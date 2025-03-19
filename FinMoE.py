import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from peft import PeftModel
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaRMSNorm
from typing import Optional, Union


class Top1Gating(nn.Module):
    """Gating network that uses the first `num_layers` of Llama as a feature extractor"""
    def __init__(self, model_id: str,
                 num_experts: int,
                 num_layers: int = 8):
        super(Top1Gating, self).__init__()

        full_model = LlamaModel.from_pretrained(model_id, output_hidden_states=True)

        ## remove extra layers from Llama model
        full_model.layers = nn.ModuleList(full_model.layers[:num_layers])
        full_model.config.num_hidden_layers = num_layers
        for param in full_model.parameters(): ## freeze Llama model
            param.requires_grad = False

        self.llama = full_model

        ## create new RMSNorm layer and classifier head
        hidden_size = full_model.config.hidden_size # C
        self.norm = LlamaRMSNorm(hidden_size, full_model.config.rms_norm_eps)
        self.classifier = nn.Linear(hidden_size, num_experts) # (C, E)
    
    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, C)

        hidden_states = self.norm(hidden_states) ## TODO: is this norm layer necessary?
        pooled = hidden_states[:, 0, :] # (B, C)

        logits = self.classifier(pooled)  # (B, E)
        probs = F.softmax(logits, dim=-1)
        return probs


def load_expert(model_id: str, expert_ckpt: str) -> PeftModel:
    base_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype="float16")
    return PeftModel.from_pretrained(base_model, expert_ckpt, torch_dtype="float16")


class FinMoE(nn.Module):
    gating_model_id = "meta-llama/Llama-3.2-1B"
    expert_model_id = "meta-llama/Llama-3.2-1B"

    def __init__(self, expert_ckpts: list[Path],
                 max_context_length: int):

        super(FinMoE, self).__init__()
        self.gate = Top1Gating(FinMoE.gating_model_id, len(expert_ckpts))

        self.experts = nn.ModuleList([load_expert(FinMoE.expert_model_id, exp_ckpt) for exp_ckpt in expert_ckpts])
        for expert in self.experts: ## freeze expert models
            for param in expert.parameters():
                param.requires_grad = False

        self.max_context_length = max_context_length
        self.vocab_size = self.experts[0].config.vocab_size
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None
    ):
        ## expert routing probs
        expert_probs = self.gate.forward(input_ids, attention_mask)  # (B, E)

        ## select top-1 expert index for each sample
        top1_indices = expert_probs.argmax(dim=-1)  # (B,)
        batch_size = expert_probs.size(0)
  
        combined_output = torch.zeros((batch_size, self.max_context_length, self.vocab_size),
                                      dtype=torch.float16,
                                      device=expert_probs.device)
        
        # For each expert, process only the samples routed to it.
        for expert_idx, expert in enumerate(self.experts):
            mask = (top1_indices == expert_idx)
            if mask.any():
                selected_input_ids = input_ids[mask]
                selected_attention = attention_mask[mask] if attention_mask is not None else None
                expert_out = expert(selected_input_ids, selected_attention)
                combined_output[mask] = expert_out.logits
        
        chosen_probs = expert_probs.gather(1, top1_indices.unsqueeze(1))  # shape (B, 1)
        logits = combined_output * chosen_probs

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

        
        return combined_output
