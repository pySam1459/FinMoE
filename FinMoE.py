import os
import torch
import torch.nn as nn
from peft import PeftModel
from transformers.configuration_utils import PretrainedConfig
from transformers.loss.loss_utils import ForCausalLMLoss, ForTokenClassification
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from typing import Optional


class FinMoEConfig(PretrainedConfig):
    """
    Config for FinMoE

    Kwargs:
        expert_ckpts (list[str]): a list of str paths to expert model checkpoints to be used in FinMoE
        model_id (str): huggingface model id used to load from remote repo
        token_list (list[int] | None): token indices used for ForTokenClassification loss
    """
    def __init__(self, **kwargs):
        super(FinMoEConfig, self).__init__(**kwargs)

        self.expert_ckpts = kwargs.pop("expert_ckpts", [])
        self.n_experts = len(self.expert_ckpts)
        self.model_id = kwargs.pop("model_id", "meta-llama/Llama-3.2-1B")
        self.gating_gaussian = kwargs.pop("gating_gaussian", 0.2)

        self.token_list = kwargs.pop("token_list", None)


class Top1Gating(nn.Module):
    """
    Gating network using a llama model for natural language feature extraction
      with a `w_gate` head for expert classification.

    Args:
        llama (LlamaModel): base llama model reference
        num_experts (int): number of expert models `w_gate` should project to
    """
    def __init__(self, config: FinMoEConfig, llama: LlamaModel):
        super(Top1Gating, self).__init__()

        self.llama = llama
        self.w_gate = nn.Linear(llama.config.hidden_size, config.n_experts) # (C, E)

        self.gaussian = config.gating_gaussian
        self.epsilon = 1e-6 # stops ZeroDivisionError when div by denominators
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_aux_loss: bool = False) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes the forward pass of the gating network

        Args:
            input_ids (Tensor): tensor of input tokens ids
            attention_mask (Tensor): tensor of attention mask, used when batch contains different length input tokens
        
        Returns:
            gate_scores (Tensor): scores for each expert for next token prediction determined by network, size (E,)
            top1_indices (Tensor): top 1 expert indices for each sample in batch, size (B,)
        """
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        gate_scores = self.w_gate(pooled)  # (B, E)
        if self.training:
            gate_scores += torch.randn_like(gate_scores) * self.gaussian

        _, top1_indices = gate_scores.topk(1, dim=-1)
        
        ## below contains tricks to make gate_scores differentiable
        ## see: https://github.com/kyegomez/SwitchTransformers/blob/main/switch_transformers/model.py
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top1_indices, 1
        )
        masked_gate_scores = gate_scores * mask
        denominators = masked_gate_scores.sum(-1, keepdim=True) + self.epsilon
        
        # Norm gate scores to sum to 1
        norm_gate_scores = masked_gate_scores / denominators

        # if use_aux_loss:
        #     print(gate_scores.shape)
        #     print(gate_scores)
        #     load = gate_scores.sum(0)  # Sum over all examples
        #     importance = gate_scores.sum(1)  # Sum over all experts

        #     # Aux loss is mean suqared difference between load and importance
        #     aux_loss = ((load - importance) ** 2).mean()

        #     return gate_scores, top1_indices.squeeze(-1), aux_loss

        return norm_gate_scores, top1_indices, None


    def save_pretrained(self, save_directory: str, **__):
        os.makedirs(save_directory, exist_ok=True)
        trainable_keys = {name
                          for name, param in self.named_parameters()
                          if param.requires_grad}

        state_dict = self.state_dict()
        trainable_state_dict = {k: v
                                for k, v in state_dict.items()
                                if k in trainable_keys}

        torch.save(trainable_state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    def load_pretrained(self, load_directory: str, **__):
        load_path = os.path.join(load_directory, "pytorch_model.bin")
        if os.path.exists(load_path):
            trainable_state_dict = torch.load(load_path, map_location="cpu", weights_only=False)
            self.load_state_dict(trainable_state_dict, strict=False) # strict=False as only some weights are loaded
        else:
            raise FileNotFoundError(f"No trainable checkpoint found at {load_path}")


class FinMoE(PreTrainedModel):
    """
    Finance Mixture of Experts

    """

    config_class = FinMoEConfig

    def __init__(self, config: FinMoEConfig):
        super(FinMoE, self).__init__(config)

        ## load base_model
        llama = LlamaForCausalLM.from_pretrained(config.model_id, torch_dtype=config.torch_dtype)

        ## Load LoRA adapters onto PeftModel
        self.n_experts = len(config.expert_ckpts)
        self.expert = PeftModel.from_pretrained(llama, config.expert_ckpts[0], adapter_name="0")
        for i, ckpt in enumerate(config.expert_ckpts[1:], start=1):
            self.expert.load_adapter(ckpt, adapter_name=str(i))

        # freeze base model and LoRA adapter params
        for param in self.expert.parameters(): 
            param.requires_grad = False

        ## pass frozen LlamaModel to Top1Gating
        self.gate = Top1Gating(config, llama.model)

        self.vocab_size = self.expert.config.vocab_size
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **loss_kwargs
    ):
        ## expert routing produces gate scores
        use_aux_loss = labels is not None
        gate_scores, top1_indices, aux_loss = self.gate.forward(input_ids, attention_mask, use_aux_loss)  # (B, E)

        combined_output = torch.zeros(input_ids.shape + (self.vocab_size, ),
                                      dtype=self.config.torch_dtype,
                                      device=self.device)
        
        batch_size = input_ids.size(0)
        # For each expert, process only the samples routed to it.
        for expert_idx in range(self.n_experts):
            mask = (top1_indices.squeeze(1) == expert_idx) # mask over batches
            if not mask.any():
                continue

            selected_input_ids = input_ids[mask]
            selected_attention = attention_mask[mask] if attention_mask is not None else None

            self.expert.set_adapter(str(expert_idx))
            expert_out = self.expert(selected_input_ids, selected_attention)
            combined_output[mask] = expert_out.logits

        logits = combined_output * gate_scores.gather(1, top1_indices).unsqueeze(-1)

        loss = None
        if labels is not None:
            if self.config.loss_type == "ForCausalLM":
                loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)

            elif self.config.loss_type == "ForTokenClassification" and self.config.token_list is not None:
                gen_idx = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(logits.size(0), device=logits.device)

                gen_logits = logits[batch_indices[:, None], gen_idx[:, None], self.config.token_list] # (B, Tok_list)

                loss_kwargs["num_items_in_batch"] = None
                loss = ForTokenClassification(logits=gen_logits, labels=labels, config=self.config, **loss_kwargs)
                # if use_aux_loss:
                #     loss += aux_loss

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