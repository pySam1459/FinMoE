import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from typing import Optional


class FinMoEConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super(FinMoEConfig, self).__init__(**kwargs)

        self.expert_ckpts = kwargs.pop("expert_ckpts", [])
        self.model_id = kwargs.pop("model_id", "meta-llama/Llama-3.2-1B")
        self.token_mask = kwargs.pop("token_mask", None)


class Top1Gating(nn.Module):
    """Gating network that uses the first `num_layers` of Llama as a feature extractor"""
    def __init__(self, llama: LlamaModel, num_experts: int):
        super(Top1Gating, self).__init__()

        self.llama = llama
        self.w_gate = nn.Linear(llama.config.hidden_size, num_experts) # (C, E)
    
    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)

        pooled = outputs.last_hidden_state[:, 0, :] # (B, C)

        logits = self.w_gate(pooled)  # (B, E)
        probs = F.softmax(logits, dim=-1)
        return probs

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
        self.token_mask = self.config.token_mask

        if self.token_mask is not None:
            inv_token_mapping = {t: i for i, t in enumerate(self.token_mask)}
            max_key = max(inv_token_mapping.keys())
            self.inv_token_lookup = torch.full((max_key + 1,), -1)
            for k, v in inv_token_mapping.items():
                self.inv_token_lookup[k] = v
            self.inv_token_lookup = self.inv_token_lookup.to(self.device)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **loss_kwargs
    ):
        ## expert routing probs
        expert_probs = self.gate.forward(input_ids, attention_mask)  # (B, E)

        ## select top-1 expert index for each sample
        # top1_indices = expert_probs.argmax(dim=-1)  # (B,)
        batch_size = expert_probs.size(0)

        context_length = input_ids.shape[1]
        vocab_size = len(self.token_mask) if self.token_mask is not None else self.vocab_size
        logits = torch.zeros((batch_size, context_length, vocab_size), # (B,T,V)
                              dtype=self.config.torch_dtype,
                              device=self.device)

        # For each expert, process only the samples routed to it.
        for expert_idx in range(self.n_experts):
            # mask = (top1_indices == expert_idx) # mask over batches
            # if not mask.any():
            #     continue

            # selected_input_ids = input_ids[mask]
            # selected_attention = attention_mask[mask] if attention_mask is not None else None

            self.expert.set_adapter(str(expert_idx))
            expert_out = self.expert(input_ids, attention_mask) # (B,T,V)

            index = torch.LongTensor([[expert_idx]]*batch_size).to(expert_probs.device) # (B, 1)
            if self.token_mask is not None:
                expert_logits = expert_out.logits[:,:,self.token_mask]
            else:
                expert_logits = expert_out.logits

            weighted_logits = expert_logits * expert_probs.gather(1, index).unsqueeze(2)
            logits += weighted_logits
            # expert_out = self.expert(selected_input_ids, selected_attention)
            # combined_output[mask] += expert_out.logits

        # chosen_probs = expert_probs.gather(1, top1_indices.unsqueeze(1))  # shape (B, 1)
        # logits = combined_output * chosen_probs

        loss = None
        if labels is not None:
            gen_idx = attention_mask.sum(dim=1).long() - 1
            pooled_logits = logits[torch.arange(logits.size(0)),gen_idx,:]
            print(labels)
            loss = self.loss_function(labels=labels, pooled_logits=pooled_logits, config=self.config)
            # loss = self.loss_function(logits=logits,
            #                           labels=labels,
            #                           vocab_size=self.vocab_size,
            #                           **loss_kwargs)

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