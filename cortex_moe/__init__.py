"""Cortex-MoE: multimodal Mixture-of-Experts model."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_cortex_moe import CortexMoeConfig
from .modeling_cortex_moe import (
    CortexMoeForCausalLM,
    CortexMoeModel,
    CortexMoeVisionTower,
)

AutoConfig.register("cortex_moe", CortexMoeConfig)
AutoModel.register(CortexMoeConfig, CortexMoeModel)
AutoModelForCausalLM.register(CortexMoeConfig, CortexMoeForCausalLM)

__all__ = [
    "CortexMoeConfig",
    "CortexMoeForCausalLM",
    "CortexMoeModel",
    "CortexMoeVisionTower",
]
