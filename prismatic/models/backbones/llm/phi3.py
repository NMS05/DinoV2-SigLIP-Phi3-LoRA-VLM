"""
Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
          (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
          (rotary_emb): Phi3RotaryEmbedding()
        )
        (mlp): Phi3MLP(
          (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): Phi3RMSNorm()
        (resid_attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
        (post_attention_layernorm): Phi3RMSNorm()
      )
    )
    (norm): Phi3RMSNorm()
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)

"""

from typing import Optional, Type

import torch
from torch import nn as nn

from transformers import Phi3ForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from peft import LoraConfig

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    Phi3ChatPromptBuilder,
)


LLM_MODELS = {
    "phi3_base": {
        "llm_family": "phi3", "llm_cls": Phi3ForCausalLM, "hf_hub_path": "microsoft/Phi-3-mini-4k-instruct"
    },
}
# fmt: on


class Phi3LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str, # phi3_base
        llm_max_length: int = 4096,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
        enable_peft = True, #      <<<<<<<<<<<<<<<         ################### set this to True for finetune stage #######################
        lora_config = LoraConfig(
            r=128, 
            lora_alpha=256, 
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "qkv_proj",
                "o_proj",
                "down_proj",
                "gate_up_proj"],
            task_type="CAUSAL_LM"
            ),
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            enable_peft = enable_peft,
            lora_peft_config = lora_config,
            **LLM_MODELS[llm_backbone_id],
        )

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return Phi3ChatPromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Phi3DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16