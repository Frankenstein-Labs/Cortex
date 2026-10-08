from transformers import PretrainedConfig


class CortexMoeConfig(PretrainedConfig):
    model_type = "cortex_moe"

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        num_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
        expert_capacity_factor=1.25,
        image_size=224,
        patch_size=16,
        vision_embed_dim=768,
        vision_depth=6,
        vision_num_heads=12,
        vision_mlp_ratio=4.0,
        max_image_tokens=64,
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.use_cache = use_cache
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.expert_capacity_factor = expert_capacity_factor
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.vision_depth = vision_depth
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.max_image_tokens = max_image_tokens

    @property
    def num_hidden_layers(self):
        return self.n_layer

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_channels(self):
        return 3
