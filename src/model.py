"""
Model loading, residual stream hooks, and activation steering.

Handles multi-GPU inference via HuggingFace accelerate device_map="auto".
Provides clean interfaces for:
  - Extracting residual stream activations at specified layers
  - Steering: adding scaled vectors to the residual stream during forward pass
  - Computing residual stream norms for calibrating steering strength

Steering strength convention (matching the Anthropic paper):
  strength=0.05 means the added vector has norm = 0.05 * mean(||residual_stream||)
  at the target layer, measured across a calibration dataset.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import gc
from loguru import logger


class ActivationCache:
    """Thread-safe cache for hooked activations."""

    def __init__(self):
        self.activations: Dict[int, torch.Tensor] = {}

    def store(self, layer_idx: int, tensor: torch.Tensor):
        # Detach and move to CPU immediately to free GPU memory
        self.activations[layer_idx] = tensor.detach().cpu()

    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.activations.get(layer_idx)

    def clear(self):
        self.activations.clear()


class SteeringConfig:
    """Configuration for a single steering intervention."""

    def __init__(
        self,
        vector: torch.Tensor,
        layer_idx: int,
        strength: float,
        residual_norm: float,
        token_mask: Optional[torch.Tensor] = None,
    ):
        self.vector = vector  # shape: (hidden_dim,)
        self.layer_idx = layer_idx
        self.strength = strength
        self.residual_norm = residual_norm
        self.token_mask = token_mask  # shape: (seq_len,) bool, True = steer this position

    @property
    def scaled_vector(self) -> torch.Tensor:
        """Vector scaled so that its norm = strength * residual_norm."""
        v = self.vector
        v_norm = v.norm()
        if v_norm < 1e-8:
            return v
        return v * (self.strength * self.residual_norm / v_norm)


class ModelWrapper:
    """
    Wraps a HuggingFace causal LM for activation extraction and steering.

    Usage:
        model = ModelWrapper("meta-llama/Llama-3.1-70B-Instruct")

        # Extract activations
        acts = model.extract_activations("Hello world", layer_indices=[20, 40, 60])
        # acts[40] has shape (1, seq_len, hidden_dim)

        # Generate with steering
        output = model.generate_steered(
            prompt="...",
            steering_configs=[SteeringConfig(vec, layer=40, strength=0.05, norm=...)],
            max_new_tokens=512,
        )
    """

    def __init__(
        self,
        model_name: str,
        torch_dtype=torch.bfloat16,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model.eval()

        # Identify transformer layers
        # Works for Llama, Qwen, Mistral, and most HF transformer models
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            self.layers = self.model.transformer.h
        else:
            raise ValueError(
                f"Cannot identify transformer layers for {model_name}. "
                "Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM."
            )

        self.num_layers = len(self.layers)
        self.hidden_dim = self.model.config.hidden_size
        self._hooks = []
        self._cache = ActivationCache()
        self._steering_hooks = []

        # Get embedding device for input placement
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            self._input_device = next(self.model.model.embed_tokens.parameters()).device
        elif hasattr(self.model, "transformer"):
            self._input_device = next(self.model.transformer.wte.parameters()).device
        else:
            self._input_device = torch.device("cuda:0")

        logger.info(
            f"Model loaded: {self.num_layers} layers, hidden_dim={self.hidden_dim}, "
            f"input_device={self._input_device}"
        )

    # ------------------------------------------------------------------
    # Activation extraction
    # ------------------------------------------------------------------

    def _make_extraction_hook(self, layer_idx: int):
        """Create a hook that captures the residual stream output of a layer."""
        def hook_fn(module, input, output):
            # output is a tuple; first element is hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            self._cache.store(layer_idx, hidden)
        return hook_fn

    def _register_extraction_hooks(self, layer_indices: List[int]):
        self._clear_hooks()
        for idx in layer_indices:
            if idx < 0 or idx >= self.num_layers:
                raise ValueError(f"Layer {idx} out of range [0, {self.num_layers})")
            hook = self.layers[idx].register_forward_hook(
                self._make_extraction_hook(idx)
            )
            self._hooks.append(hook)

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._cache.clear()

    def _clear_steering_hooks(self):
        for h in self._steering_hooks:
            h.remove()
        self._steering_hooks.clear()

    def _tokenize(self, text: str, add_special_tokens: bool = True) -> dict:
        """Tokenize and move to the correct device."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        return {k: v.to(self._input_device) for k, v in inputs.items()}

    @torch.no_grad()
    def extract_activations(
        self,
        text: str,
        layer_indices: List[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Run forward pass and return residual stream activations at specified layers.

        Returns:
            Dict mapping layer_idx -> tensor of shape (1, seq_len, hidden_dim) on CPU.
        """
        self._register_extraction_hooks(layer_indices)
        inputs = self._tokenize(text)
        self.model(**inputs)
        result = {k: v.clone() for k, v in self._cache.activations.items()}
        self._clear_hooks()
        return result

    @torch.no_grad()
    def extract_mean_activations(
        self,
        text: str,
        layer_indices: List[int],
        token_offset: int = 50,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations and return the MEAN across token positions,
        starting from token_offset (matching the paper's methodology).

        Returns:
            Dict mapping layer_idx -> tensor of shape (hidden_dim,) on CPU.
        """
        activations = self.extract_activations(text, layer_indices)
        result = {}
        for idx, act in activations.items():
            # act shape: (1, seq_len, hidden_dim)
            seq_len = act.shape[1]
            start = min(token_offset, seq_len - 1)  # Safety: don't exceed seq length
            if start >= seq_len - 1:
                # Story too short; use all tokens (with a warning)
                logger.warning(
                    f"Text has only {seq_len} tokens, less than offset {token_offset}. "
                    f"Using all tokens."
                )
                start = 0
            result[idx] = act[0, start:, :].mean(dim=0)  # (hidden_dim,)
        return result

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    def _make_steering_hook(self, config: SteeringConfig):
        """Create a hook that adds a steering vector to the residual stream."""
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            device = hidden.device
            sv = config.scaled_vector.to(device)  # (hidden_dim,)

            if config.token_mask is not None:
                mask = config.token_mask.to(device)
                # Expand mask: (seq_len,) -> (1, seq_len, 1)
                # Handle case where mask is shorter than hidden (new tokens in generation)
                actual_len = hidden.shape[1]
                if mask.shape[0] < actual_len:
                    # During generation, new tokens appear. Steer all of them.
                    extended = torch.ones(actual_len, device=device, dtype=mask.dtype)
                    extended[:mask.shape[0]] = mask
                    mask = extended
                elif mask.shape[0] > actual_len:
                    mask = mask[:actual_len]
                mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
                hidden = hidden + sv.unsqueeze(0).unsqueeze(0) * mask
            else:
                hidden = hidden + sv.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook_fn

    def _register_steering_hooks(self, configs: List[SteeringConfig]):
        self._clear_steering_hooks()
        for config in configs:
            hook = self.layers[config.layer_idx].register_forward_hook(
                self._make_steering_hook(config)
            )
            self._steering_hooks.append(hook)

    @torch.no_grad()
    def generate_steered(
        self,
        prompt: str,
        steering_configs: Optional[List[SteeringConfig]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        extract_layers: Optional[List[int]] = None,
    ) -> dict:
        """
        Generate text with optional steering and optional activation extraction.

        Args:
            prompt: Input text.
            steering_configs: List of SteeringConfig for interventions.
            extract_layers: If provided, also extract activations at these layers
                           (on the FULL sequence including generated tokens).

        Returns:
            dict with keys:
                "text": generated text (response only, prompt stripped)
                "full_text": full text including prompt
                "input_ids": token IDs of the full sequence
                "activations": dict of layer -> activations (if extract_layers given)
        """
        if steering_configs:
            self._register_steering_hooks(steering_configs)

        if extract_layers:
            self._register_extraction_hooks(extract_layers)

        inputs = self._tokenize(prompt)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        # IMPORTANT: Clear steering hooks BEFORE extraction forward pass.
        # We want V_internal to reflect the model's natural representation
        # of the generated text, not the artificially steered state.
        # This is critical for faithfulness analysis: we're asking whether
        # the TEXT the model produced carries the emotional signal, not
        # whether the steering vector we injected is detectable.
        self._clear_steering_hooks()

        # If we need activations on the full output, do another forward pass
        activations = {}
        if extract_layers:
            self._clear_hooks()
            self._register_extraction_hooks(extract_layers)
            full_input = {"input_ids": output_ids.to(self._input_device)}
            self.model(**full_input)
            activations = {k: v.clone() for k, v in self._cache.activations.items()}
            self._clear_hooks()

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response_text = self.tokenizer.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        )

        return {
            "text": response_text,
            "full_text": full_text,
            "input_ids": output_ids[0].cpu(),
            "prompt_len": prompt_len,
            "activations": activations,
        }

    @torch.no_grad()
    def generate_plain(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """Simple generation without steering or activation extraction."""
        inputs = self._tokenize(prompt)
        prompt_len = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Residual stream norm calibration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_residual_norms(
        self,
        calibration_texts: List[str],
        layer_indices: List[int],
    ) -> Dict[int, float]:
        """
        Compute mean residual stream norm at each layer, across calibration texts.
        Used to calibrate steering strength.

        Returns:
            Dict mapping layer_idx -> mean L2 norm (float).
        """
        norms = {idx: [] for idx in layer_indices}
        for text in calibration_texts:
            acts = self.extract_activations(text, layer_indices)
            for idx, act in acts.items():
                # act shape: (1, seq_len, hidden_dim)
                layer_norm = act.norm(dim=-1).mean().item()  # mean over batch & seq
                norms[idx].append(layer_norm)

        return {idx: sum(vals) / len(vals) for idx, vals in norms.items()}

    # ------------------------------------------------------------------
    # Logit lens
    # ------------------------------------------------------------------

    @torch.no_grad()
    def logit_lens(
        self,
        vector: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Project a vector through the unembedding matrix to see which tokens
        it upweights/downweights.

        Returns:
            (top_tokens, bottom_tokens): each is a list of (token_str, logit_value).
        """
        # Get the unembedding (lm_head) weight
        if hasattr(self.model, "lm_head"):
            unembed = self.model.lm_head.weight  # (vocab_size, hidden_dim)
        else:
            raise ValueError("Cannot find lm_head for logit lens")

        device = unembed.device
        v = vector.to(device).float()
        logits = unembed.float() @ v  # (vocab_size,)

        top_indices = logits.topk(top_k).indices
        bottom_indices = logits.topk(top_k, largest=False).indices

        top_tokens = [
            (self.tokenizer.decode([idx.item()]), logits[idx].item())
            for idx in top_indices
        ]
        bottom_tokens = [
            (self.tokenizer.decode([idx.item()]), logits[idx].item())
            for idx in bottom_indices
        ]

        return top_tokens, bottom_tokens

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Free GPU memory."""
        self._clear_hooks()
        self._clear_steering_hooks()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
