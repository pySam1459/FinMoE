import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from transformers.loss.loss_utils import ForCausalLMLoss, ForTokenClassification
from typing import Optional


class FinMoEConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super(FinMoEConfig, self).__init__(**kwargs)

        self.expert_ckpts = kwargs.pop("expert_ckpts", [])
        self.model_id = kwargs.pop("model_id", "meta-llama/Llama-3.2-1B")
        self.n_gate_layers = kwargs.pop("n_gate_layers", 16)

        self.token_list = kwargs.pop("token_list", None)


class Top1Gating(nn.Module):
    """Gating network that uses the first `num_layers` of Llama as a feature extractor"""
    def __init__(self, llama: LlamaModel, num_experts: int):
        super(Top1Gating, self).__init__()

        self.llama = llama
        self.w_gate = nn.Linear(llama.config.hidden_size, num_experts) # (C, E)

        self.epsilon = 1e-6
    
    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        gate_scores = self.w_gate(pooled)  # (B, E)

        _, top1_indices = gate_scores.topk(1, dim=-1)
        
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top1_indices, 1
        )
        masked_gate_scores = gate_scores * mask
        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        
        # Norm gate scores to sum to 1
        gate_scores = masked_gate_scores / denominators

        return gate_scores, top1_indices.squeeze(0)


    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        trainable_keys = {name
                          for name, param in self.named_parameters()
                          if param.requires_grad}

        state_dict = self.state_dict()
        trainable_state_dict = {k: v
                                for k, v in state_dict.items()
                                if k in trainable_keys}

        torch.save(trainable_state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    def load_pretrained(self, load_directory: str, **kwargs):
        load_path = os.path.join(load_directory, "pytorch_model.bin")
        if os.path.exists(load_path):
            trainable_state_dict = torch.load(load_path, map_location="cpu", weights_only=False)
            self.load_state_dict(trainable_state_dict, strict=False) # strict=False as only some weights are loaded
        else:
            raise FileNotFoundError(f"No trainable checkpoint found at {load_path}")


class FinMoE(PreTrainedModel):
    config_class = FinMoEConfig

    def __init__(self, config: FinMoEConfig):
        super(FinMoE, self).__init__(config)

        ## load base_model
        llama = LlamaForCausalLM.from_pretrained(config.model_id,
                                                 output_hidden_states=True,
                                                 torch_dtype=config.torch_dtype)

        ## Load LoRA adapters onto PeftModel
        self.n_experts = len(config.expert_ckpts)
        self.expert = PeftModel.from_pretrained(llama, config.expert_ckpts[0], adapter_name="0")
        for i, ckpt in enumerate(config.expert_ckpts[1:], start=1):
            self.expert.load_adapter(ckpt, adapter_name=str(i))

        # freeze base model and LoRA adapter params
        for param in self.expert.parameters(): 
            param.requires_grad = False

        ## pass LlamaModel to Top1Gating
        self.gate = Top1Gating(llama.model, self.n_experts)

        self.vocab_size = self.expert.config.vocab_size
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **loss_kwargs
    ):
        ## expert routing produces gate scores
        gate_scores, top1_indices = self.gate.forward(input_ids, attention_mask)  # (B, E)

        combined_output = torch.zeros(input_ids.shape + (self.vocab_size, ),
                                      dtype=self.config.torch_dtype,
                                      device=self.device)
        
        # For each expert, process only the samples routed to it.
        for expert_idx in range(self.n_experts):
            mask = (top1_indices == expert_idx) # mask over batches
            if not mask.any():
                continue

            selected_input_ids = input_ids[mask]
            selected_attention = attention_mask[mask] if attention_mask is not None else None

            self.expert.set_adapter(str(expert_idx))
            expert_out = self.expert(selected_input_ids, selected_attention)
            combined_output[mask] = expert_out.logits

        logits = combined_output * gate_scores.gather(1, top1_indices.unsqueeze(0))

        loss = None
        if labels is not None:
            if self.config.loss_type == "ForCausalLM":
                loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)

            elif self.config.loss_type == "ForTokenClassification" and self.config.token_list is not None:
                gen_idx = attention_mask.sum(dim=1).long() - 1
                gen_logits = logits[torch.arange(logits.size(0), device=logits.device), gen_idx, self.config.token_list] # (B, Tok_list)

                loss_kwargs["num_items_in_batch"] = None
                loss = ForTokenClassification(logits=gen_logits, labels=labels, config=self.config, **loss_kwargs)
            else:
                raise ValueError(f"{self.config.loss_type} is not implemented or missing config arguments")

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        # Determine trainable parameter keys in the entire model
        trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
        state_dict = self.state_dict()
        trainable_state_dict = {k: v for k, v in state_dict.items() if k in trainable_keys}

        torch.save(trainable_state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)

        gate_save_dir = os.path.join(save_directory, "gate")
        self.gate.save_pretrained(gate_save_dir)

    @classmethod
    def load_pretrained(cls, load_directory: str, **kwargs):
        config = PretrainedConfig.from_pretrained(load_directory)
        model = cls(config)
        
        # Load the trainable parameters for FinMoE
        trainable_path = os.path.join(load_directory, "pytorch_model.bin")
        if os.path.exists(trainable_path):
            trainable_state_dict = torch.load(trainable_path, map_location="cpu", weights_only=False)
            model.load_state_dict(trainable_state_dict, strict=False)  # strict=False as only some weights are loaded
        else:
            raise FileNotFoundError(f"No trainable checkpoint found at {trainable_path}")
        
        # Load the gating network's trainable parameters from its subdirectory
        gate_dir = os.path.join(load_directory, "gate")
        model.gate.load_pretrained(gate_dir)
        
        return model