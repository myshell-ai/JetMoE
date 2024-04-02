""" JetMoE model configuration"""
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
from transformers.utils import logging
import torch.nn.init as init
import json

logger = logging.get_logger(__name__)


class JetMoEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JetMoEModel`]. It is used to instantiate a
    JetMoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the JetMoE
    [jetmoe-small](https://huggingface.co/jetmoe-small) architecture. Configuration objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50400):
            Vocabulary size of the JetMoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JetMoEModel`].
        n_positions (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        rotary_dim (`int`, *optional*, defaults to 64):
            Number of dimensions in the embedding that Rotary Position Embedding is applied to.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import JetMoEConfig, JetMoEModel

    >>> # Initializing a JetMoE 6B configuration
    >>> configuration = JetMoEConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = JetMoEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "jetmoe"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50295,
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        kv_channels = 128,
        ffn_hidden_size=2048,
        max_position_embeddings=4096,
        rotary_percent=1.0,
        activation_function="silu",
        glu=True,
        moe_num_experts=8,
        moe_top_k=2,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        bias=True,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        initializer_range=0.01,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels
        self.ffn_hidden_size = ffn_hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rotary_percent = rotary_percent
        self.activation_function = activation_function
        self.glu = glu
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.init_method = init.xavier_uniform_
        self.output_layer_init_method = init.xavier_uniform_
        self.bias = bias
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

    def to_dict(self):
        """Returns a dictionary representation of the config, excluding non-serializable attributes."""
        return {k: v for k, v in self.__dict__.items() if k not in ['init_method', 'output_layer_init_method', 'torch_dtype', '_pre_quantization_dtype', 'quantization_config']}

    def to_json_string(self, use_diff=False):
        """Serializes this instance to a JSON string, excluding non-serializable attributes.
        
        Args:
            use_diff (bool): Whether to use differences with the default config. This argument is
                             accepted for compatibility with the transformers library but is not
                             used in this custom implementation.
        """
        config_dict = self.to_dict()  # Assuming you have a to_dict method as shown earlier
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

class JetMoEOnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        """
        Initialize the JetMoEOnnxConfig.

        Args:
            config (PretrainedConfig): Pretrained model configuration.
            task (str): Task description.
            patching_specs (List[PatchingSpec]): List of patching specifications.
            use_past (bool): Whether to use past tokens in the configuration.
        """
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Define the input mappings.

        Returns:
            Mapping[str, Mapping[int, str]]: Input mappings.
        """
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_layers(self) -> int:
        """
        Get the number of layers.

        Returns:
            int: Number of layers.
        """
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        """
        Get the number of attention heads.

        Returns:
            int: Number of attention heads.
        """
        return self._config.n_head

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        """
        Generate dummy inputs for testing.

        Args:
            tokenizer (PreTrainedTokenizer): Pretrained tokenizer.
            batch_size (int): Batch size.
            seq_length (int): Sequence length.
            is_pair (bool): Whether the input is a pair.
            framework (Optional[TensorType]): Tensor framework.

        Returns:
            Mapping[str, Any]: Dummy inputs.
        """
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        """
        Get the default ONNX opset version.

        Returns:
            int: Default ONNX opset version.
        """
        return 13
