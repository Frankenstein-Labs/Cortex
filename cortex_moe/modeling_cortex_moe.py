import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_cortex_moe import CortexMoeConfig

logger = logging.get_logger(__name__)


class _Conv1D(nn.Module):
    """GPT-2 style 1D convolution: weight layout is [in, out]."""

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


def gelu_new(x: Tensor) -> Tensor:
    return F.gelu(x, approximate="tanh")


ACT_FNS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "silu": F.silu,
}


class CortexMoeAttention(nn.Module):
    def __init__(self, config: CortexMoeConfig, is_cross_attention=False):
        super().__init__()
        max_positions = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scale_attn_weights = True
        self.is_cross_attention = is_cross_attention
        if is_cross_attention:
            self.c_attn = _Conv1D(3 * self.embed_dim, self.embed_dim)
            self.q_attn = _Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = _Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = _Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_values=None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `CortexMoeAttention(..., is_cross_attention=True)`."
                )
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if past_key_values is not None and encoder_hidden_states is None:
            key, value = past_key_values.update(key, value, layer_idx)

        present = past_key_values if use_cache else None

        if self.bias is not None and query.size(-2) == key.size(-2):
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            min_value = torch.finfo(query.dtype).min
            additive = torch.zeros_like(causal_mask, dtype=query.dtype).masked_fill(
                ~causal_mask, min_value
            )
            if attention_mask is not None:
                additive = additive + (1.0 - attention_mask.to(query.dtype)) * min_value
            attention_mask = additive

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class CortexMoeMLP(nn.Module):
    def __init__(self, intermediate_size, config: CortexMoeConfig):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = _Conv1D(intermediate_size, embed_dim)
        self.c_proj = _Conv1D(embed_dim, intermediate_size)
        self.act = ACT_FNS[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CortexMoeRouter(nn.Module):
    def __init__(self, config: CortexMoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.aux_loss_coef = config.router_aux_loss_coef
        self.z_loss_coef = config.router_z_loss_coef
        self.capacity_factor = config.expert_capacity_factor
        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)

    def forward(self, hidden_states: Tensor):
        batch_size, seq_len, embed_dim = hidden_states.shape
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        aux_loss = None
        z_loss = None
        if self.training:
            scores = routing_weights
            num_tokens = scores.shape[0] * scores.shape[1]
            expert_mask = F.one_hot(topk_ids, num_classes=self.num_experts).sum(dim=-2)
            fraction = scores.mean(dim=(0, 1))
            tokens_per_expert = expert_mask.float().mean(dim=(0, 1))
            aux_loss = self.num_experts * (fraction * tokens_per_expert).sum()
            aux_loss = aux_loss * self.aux_loss_coef
            z_loss = (torch.logsumexp(router_logits, dim=-1) ** 2).mean() * self.z_loss_coef
        return router_logits, topk_weights, topk_ids, aux_loss, z_loss


class CortexMoeSparseMLP(nn.Module):
    def __init__(self, config: CortexMoeConfig):
        super().__init__()
        intermediate_size = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.capacity_factor = config.expert_capacity_factor
        self.router = CortexMoeRouter(config)
        self.experts = nn.ModuleList(
            [CortexMoeMLP(intermediate_size, config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: Tensor):
        batch_size, seq_len, embed_dim = hidden_states.shape
        router_logits, topk_weights, topk_ids, aux_loss, z_loss = self.router(hidden_states)
        flat_hidden = hidden_states.reshape(-1, embed_dim)
        flat_topk_ids = topk_ids.reshape(-1, self.top_k)
        flat_topk_weights = topk_weights.reshape(-1, self.top_k)
        num_tokens = flat_hidden.shape[0]
        capacity = min(
            int(self.capacity_factor * self.top_k * num_tokens / self.num_experts) + 1,
            num_tokens * self.top_k,
        )
        expert_mask = F.one_hot(flat_topk_ids, num_classes=self.num_experts)
        y = torch.zeros_like(flat_hidden)
        for expert_idx in range(self.num_experts):
            token_per_expert = expert_mask[:, :, expert_idx].sum(dim=-1) > 0
            tokens_for_expert = torch.nonzero(token_per_expert, as_tuple=False).squeeze(-1)
            if tokens_for_expert.numel() == 0:
                continue
            selected_tokens = tokens_for_expert[:capacity]
            expert_input = flat_hidden[selected_tokens]
            expert_output = self.experts[expert_idx](expert_input)
            weights = torch.zeros_like(selected_tokens, dtype=torch.float32)
            for k in range(self.top_k):
                mask_k = flat_topk_ids[selected_tokens, k] == expert_idx
                weights[mask_k] = flat_topk_weights[selected_tokens, k][mask_k]
            y.index_add_(0, selected_tokens, expert_output * weights.unsqueeze(-1))
        y = y.reshape(batch_size, seq_len, embed_dim)
        return y, router_logits, aux_loss, z_loss


class CortexMoeBlock(nn.Module):
    def __init__(self, config: CortexMoeConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CortexMoeAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.moe = CortexMoeSparseMLP(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_values=None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        moe_output, router_logits, aux_loss, z_loss = self.moe(hidden_states)
        hidden_states = residual + moe_output
        outputs = (hidden_states,) + attn_outputs[1:]
        outputs = outputs + (router_logits, aux_loss, z_loss)
        return outputs


class CortexMoePreTrainedModel(PreTrainedModel):
    config_class = CortexMoeConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CortexMoeBlock"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CortexMoeModel(CortexMoePreTrainedModel):
    def __init__(self, config: CortexMoeConfig):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CortexMoeBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            past_length = past_key_values.get_seq_length()
        elif (
            isinstance(past_key_values, (tuple, list))
            and len(past_key_values) > 0
            and past_key_values[0] is not None
        ):
            past_length = past_key_values[0][0].size(-2)
        else:
            past_length = 0

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            token_type_embeds = token_type_embeds.to(inputs_embeds.dtype)
        else:
            token_type_embeds = 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + input_shape[-1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = past_key_values if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        aux_losses = []
        z_losses = []
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                past_key_values=past_key_values if use_cache else None,
                layer_idx=i,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            aux_losses.append(outputs[-2])
            z_losses.append(outputs[-1])

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        aux_loss = None
        z_loss = None
        if self.training:
            valid_aux = [l for l in aux_losses if l is not None]
            valid_z = [l for l in z_losses if l is not None]
            if valid_aux:
                aux_loss = torch.stack(valid_aux).mean()
            if valid_z:
                z_loss = torch.stack(valid_z).mean()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, aux_loss, z_loss]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CortexMoeVisionTower(nn.Module):
    def __init__(self, config: CortexMoeConfig):
        super().__init__()
        self.config = config
        embed_dim = config.vision_embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.blocks = nn.ModuleList()
        for _ in range(config.vision_depth):
            self.blocks.append(self._build_block(embed_dim, config))
        self.ln_post = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)
        self.proj = nn.Linear(embed_dim, config.n_embd)
        self.max_image_tokens = config.max_image_tokens
        self._init_vision_weights()

    def _build_block(self, embed_dim, config):
        block = nn.ModuleDict({})
        block["ln_1"] = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)
        block["attn"] = nn.MultiheadAttention(
            embed_dim, config.vision_num_heads, batch_first=True, dropout=0.0
        )
        block["ln_2"] = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)
        mlp_hidden = int(embed_dim * config.vision_mlp_ratio)
        block["mlp"] = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        return block

    def _init_vision_weights(self):
        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            h = block["ln_1"](x)
            attn_out, _ = block["attn"](h, h, h, need_weights=False)
            x = x + attn_out
            x = x + block["mlp"](block["ln_2"](x))
        x = self.ln_post(x)
        x = self.proj(x)
        x = x[:, 1:]
        if x.shape[1] > self.max_image_tokens:
            stride = math.ceil(x.shape[1] / self.max_image_tokens)
            x = x[:, ::stride]
        return x


class CortexMoeVisionProjector(nn.Module):
    def __init__(self, config: CortexMoeConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.vision_embed_dim, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

    def forward(self, vision_features: Tensor) -> Tensor:
        return self.proj(vision_features)


class CortexMoeForCausalLM(CortexMoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config: CortexMoeConfig):
        super().__init__(config)
        self.config = config
        self.transformer = CortexMoeModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.vision_tower = CortexMoeVisionTower(config)
        self.vision_projector = CortexMoeVisionProjector(config)
        self.max_image_tokens = config.max_image_tokens
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.transformer.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else True
        vision_tokens = None
        if pixel_values is not None:
            vision_features = self.vision_tower(pixel_values)
            vision_tokens = self.vision_projector(vision_features)
            if input_ids is not None:
                prefix_len = vision_tokens.shape[1]
                total_len = prefix_len + input_ids.shape[1]
                prefix_mask = torch.ones(
                    (input_ids.shape[0], prefix_len),
                    dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                    device=input_ids.device,
                )
                attention_mask = (
                    torch.cat([prefix_mask, attention_mask], dim=1)
                    if attention_mask is not None
                    else prefix_mask
                )
        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        if vision_tokens is not None:
            inputs_embeds = torch.cat([vision_tokens.to(inputs_embeds.dtype), inputs_embeds], dim=1)
            if position_ids is None:
                position_ids = torch.arange(
                    inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0)

        transformer_outputs = self.transformer(
            input_ids=None if inputs_embeds is not None else input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif (
                isinstance(past_key_values, (tuple, list))
                and len(past_key_values) > 0
                and past_key_values[0] is not None
            ):
                past_length = past_key_values[0][0].size(-2)
            else:
                past_length = 0
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs = {"input_ids": input_ids}
            if pixel_values is not None and past_key_values is None:
                model_inputs["pixel_values"] = pixel_values
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
