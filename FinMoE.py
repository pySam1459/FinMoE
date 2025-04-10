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
        self.w_gate = nn.Linear(llama.config.hidden_size, config.n_experts, bias=False) # (C, E)

        self.gaussian = config.gating_gaussian
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes the forward pass of the gating network

        Args:
            input_ids (Tensor): tensor of input tokens ids
            attention_mask (Tensor): tensor of attention mask, used when batch contains different length input tokens
        
        Returns:
            gate_scores (Tensor): scores for each expert for next token prediction determined by network, size (E,)
        """
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.w_gate(outputs.last_hidden_state)# (B, T, E)
        
        batch_size = input_ids.shape[0]
        gen_idx = attention_mask.sum(dim=1).long() - 1
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), gen_idx] # (B, E)
        # if self.training:
        #     gate_scores += torch.randn_like(gate_scores) * self.gaussian

        return torch.softmax(pooled_logits, dim=-1)


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
        gate_scores = self.gate.forward(input_ids, attention_mask)  # (B, E)

        logits = torch.zeros(input_ids.shape + (self.vocab_size, ),
                             dtype=self.config.torch_dtype,
                             device=self.device)
        
        # For each expert, process only the samples routed to it.
        batch_size = input_ids.size(0)
        for expert_idx in range(self.n_experts):
            self.expert.set_adapter(str(expert_idx))

            expert_out = self.expert(input_ids, attention_mask)
            logits += expert_out.logits * gate_scores[:, expert_idx].view(batch_size, 1, 1)

        loss = None
        if labels is not None:
            if self.config.loss_type == "ForCausalLM":
                loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)

            elif self.config.loss_type == "ForTokenClassification" and self.config.token_list is not None:
                gen_idx = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(logits.size(0), device=logits.device)

                gen_logits = logits[batch_indices[:, None], gen_idx[:, None], self.config.token_list] # (B, Tok_list)

                loss_kwargs["num_items_in_batch"] = None ## default = 0, causing ZeroDivisionError
                loss = ForTokenClassification(logits=gen_logits, labels=labels, config=self.config, **loss_kwargs)

            else:
                raise ValueError(f"{self.config.loss_type} is not implemented or missing config arguments")

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    def predict(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...

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
