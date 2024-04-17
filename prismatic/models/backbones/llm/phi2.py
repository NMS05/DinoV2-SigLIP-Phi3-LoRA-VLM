from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import PhiForCausalLM
from transformers.models.phi.modeling_phi import PhiDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    PurePromptBuilder,
)


LLM_MODELS = {
    "phi2_base": {
        "llm_family": "phi2", "llm_cls": PhiForCausalLM, "hf_hub_path": "microsoft/phi-2"
    },
}
# fmt: on


class Phi2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str, # phi2base
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **LLM_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm.resize_token_embeddings(len(self.tokenizer))

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return PhiDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
