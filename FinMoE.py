# FinMoE implementation by Samuel Barnett
# Submitted as part of the degree of MEng Computer Science 
#   to the Board of Examiners in the Department of Computer Sciences, Durham University
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
        g_net_id (str): Name of gating subclass to use
        token_list (list[int] | None): token indices used for ForTokenClassification loss
    """
    def __init__(self, **kwargs):
        super(FinMoEConfig, self).__init__(**kwargs)

        self.expert_ckpts = kwargs.pop("expert_ckpts", [])
        self.n_experts = len(self.expert_ckpts)

        self.model_id = kwargs.pop("model_id", "meta-llama/Llama-3.2-1B")
        self.g_net_id = kwargs.pop("g_net_id", "LlamaGating")
        self.topk = kwargs.pop("topk", "top3")

        self.token_list = kwargs.pop("token_list", None)


class GatingBase(nn.Module):
    """
    Base class for the different Gating Network variants
    Includes methods `save_pretrained` and `load_pretrained`
    """
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        ...

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


class FastGating(GatingBase):
    """
    Gating network using LogSumExp over Llama model input embeddings for natural language feature extraction
      with a `w_gate` head for expert classification.

    Args:
        llama (LlamaModel): base llama model reference
        num_experts (int): number of expert models `w_gate` should project to
    """
    def __init__(self, config: FinMoEConfig, llama: LlamaModel):
        super(FastGating, self).__init__()

        self.embed_tokens = llama.embed_tokens
        self.w_gate = nn.Linear(llama.config.hidden_size, config.n_experts, bias=False) # (C, E)

        self.temperature = 1.0
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Computes the forward pass of the FastGating network

        Args:
            input_ids (Tensor): tensor of input tokens ids
            attention_mask (Tensor): tensor of attention mask, used when batch contains different length input tokens
        Returns:
            gate_scores (Tensor): scores for each expert for next token prediction determined by network, size (E,)
        """
        # compute logits using llama embeddings and w_gate
        embds = self.embed_tokens(input_ids) # (B, T, C)
        logits = self.w_gate(embds) # (B, T, E)

        # fix to RuntimeError caused by in-place operations modifying variables needed for gradient computations
        logits_scaled = logits / self.temperature

        # mask out logits
        mask_expanded = attention_mask.bool().unsqueeze(-1)
        logits_scaled_masked = logits_scaled.masked_fill(~mask_expanded, float('-inf')) # (B, T, E)

        pooled_logits = self.temperature * torch.logsumexp(logits_scaled_masked, dim=1) # (B, E)
        return torch.softmax(pooled_logits, dim=-1)


class LlamaGating(GatingBase):
    """
    Gating network using a llama model for natural language feature extraction
      with a `w_gate` head for expert classification.

    Args:
        llama (LlamaModel): base llama model reference
        num_experts (int): number of expert models `w_gate` should project to
    """
    def __init__(self, config: FinMoEConfig, llama: LlamaModel):
        super(LlamaGating, self).__init__()

        self.llama = llama
        self.w_gate = nn.Linear(llama.config.hidden_size, config.n_experts, bias=False) # (C, E)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Computes the forward pass of the LlamaGating network

        Args:
            input_ids (Tensor): tensor of input tokens ids
            attention_mask (Tensor): tensor of attention mask, used when batch contains different length input tokens
        
        Returns:
            gate_scores (Tensor): scores for each expert for next token prediction determined by network, size (E,)
        """
        # compute logits using llama forward pass and w_gate
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.w_gate(outputs.last_hidden_state) # (B, T, E)

        # select logit repr'ing token to be generated, across batch
        batch_size = input_ids.shape[0]
        gen_idx = attention_mask.sum(dim=1).long() - 1  # index of token to be generated
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), gen_idx] # (B, E)

        return torch.softmax(pooled_logits, dim=-1)


class FinMoE(PreTrainedModel):
    """Mixture of Experts Finance Language Model"""

    config_class = FinMoEConfig
    gating_networks = dict[str, GatingBase]({
        "FastGating": FastGating,
        "LlamaGating": LlamaGating
    })

    def __init__(self, config: FinMoEConfig):
        super(FinMoE, self).__init__(config)

        # load base_model
        llama = LlamaForCausalLM.from_pretrained(config.model_id, torch_dtype=config.torch_dtype)

        # Load LoRA adapters onto new PeftModel
        self.n_experts = config.n_experts
        self.expert = PeftModel.from_pretrained(llama, config.expert_ckpts[0], adapter_name="0")
        for i, ckpt in enumerate(config.expert_ckpts[1:], start=1):
            self.expert.load_adapter(ckpt, adapter_name=str(i))

        # freeze base model and LoRA adapter params
        for param in self.expert.parameters(): 
            param.requires_grad = False

        # pass frozen LlamaModel to Top3Gating
        if config.g_net_id in FinMoE.gating_networks:
            self.gate = FinMoE.gating_networks[config.g_net_id](config, llama.model)
        else:
            avail_networks = ", ".join(FinMoE.gating_networks.keys())
            raise ValueError(f"FinMoEConfig g_net_id {config.g_net_id} is invalid. Available gating networks: {avail_networks}")

        # pre-determine topk expert selection function
        if config.topk == "top1":
            self._expert_loop = self._expert_loop_top1
        elif config.topk == "top3":
            self._expert_loop = self._expert_loop_top3
        else:
            raise ValueError(f"FinMoEConfig topk {config.topk} is invalid. Avaialble topk: top1, top3")

        self.vocab_size = self.expert.config.vocab_size # V
        self.epsilon = 1e-6 # stops ZeroDivisionError when div by denominators, used in top1 loop
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **loss_kwargs
    ):
        """
        Computes the logits using top-3 expert selection.
        Output is computed from all experts and weighted by gate_scores to produce output logits

        Args:
            input_ids (Tensor): input sequence of token ids
            attention_mask (Tensor | None): used to mask padding tokens when batch training with samples of different lengths
            labels (Tensor | None): 
        """
        # expert routing produces gate scores
        gate_scores = self.gate.forward(input_ids, attention_mask)  # (B, E)

        # compute logits via top-1 or top-3 expert loops
        logits = self._expert_loop(gate_scores, input_ids, attention_mask)

        loss = None
        if labels is not None:
            # Compute loss if labels are provided
            loss = self._compute_loss(logits, labels, attention_mask, **loss_kwargs)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    def _expert_loop_top3(self,
                          gate_scores: torch.FloatTensor,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.Tensor]) -> torch.FloatTensor:
        """
        Computes the logits using top-3 expert selection.
        Output is computed from all experts and weighted by gate_scores to produce output logits

        Args:
            gate_scores (Tensor): probability distribution computed by gating network for expert routes
            input_ids (Tensor): input sequence of token ids
            attention_mask (Tensor): used to mask padding tokens when batch training with samples of different lengths
        """
        logits = torch.zeros(input_ids.shape + (self.vocab_size, ),
                             dtype=self.config.torch_dtype,
                             device=self.device)
        
        # For each expert, process only the samples routed to it.
        batch_size = input_ids.size(0)
        for expert_idx in range(self.n_experts):
            self.expert.set_adapter(str(expert_idx))

            # pass inputs through expert and multiple by gating scores
            expert_out = self.expert(input_ids, attention_mask) # (B, T, V)
            logits += expert_out.logits * gate_scores[:, expert_idx].view(batch_size, 1, 1)
        
        return logits
    
    def _expert_loop_top1(self,
                          gate_scores: torch.FloatTensor,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.Tensor]) -> None:
        """
        Computes the logits using top-1 expert selection.
        Output is computed by selecting the expert with the maximum gate score, and returning its output.

        Args:
            gate_scores (Tensor): probability distribution computed by gating network for expert routes
            input_ids (Tensor): input sequence of token ids
            attention_mask (Tensor): used to mask padding tokens when batch training with samples of different lengths
        """
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

        combined_output = torch.zeros(input_ids.shape + (self.vocab_size, ),
                                      dtype=self.config.torch_dtype,
                                      device=self.device)
        
        # For each expert, process only the samples routed to it.
        for expert_idx in range(self.n_experts):
            mask = (top1_indices.squeeze(1) == expert_idx) # mask over batches
            if not mask.any():
                # expert was not selected across entire batch
                continue
            
            # input_ids and attention masks for seleted samples in batch
            selected_input_ids = input_ids[mask]
            selected_attn_mask = attention_mask[mask] if attention_mask is not None else None

            self.expert.set_adapter(str(expert_idx))
            expert_out = self.expert(selected_input_ids, selected_attn_mask)
            combined_output[mask] = expert_out.logits

        # logits = combined_outputs with gate_scores connection
        return combined_output * norm_gate_scores.gather(1, top1_indices).unsqueeze(-1)

    def _compute_loss(self,
                      logits: torch.FloatTensor,
                      labels: torch.LongTensor,
                      attention_mask: Optional[torch.Tensor],
                      **loss_kwargs) -> torch.Tensor:
        """
        Computes the loss for CausalLM or TokenClassificatoin
        """
        if self.config.loss_type == "ForCausalLM":
            return ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)

        elif self.config.loss_type == "ForTokenClassification" and self.config.token_list is not None:
            if attention_mask is None:
                raise ValueError("Attention mask is required to select last token in sequence")

            # select last token index in input sequence
            gen_idx = attention_mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(logits.size(0), device=logits.device)

            # (B, T, V) -> (B, Tok_list)
            # T dim is squeezed by indexing to the last token `gen_idx`
            # Tokens from `self.config.token_list` are selected from V dim
            gen_logits = logits[batch_indices[:, None], gen_idx[:, None], self.config.token_list] # (B, Tok_list)

            loss_kwargs["num_items_in_batch"] = None ## default = 0, causing ZeroDivisionError
            return ForTokenClassification(logits=gen_logits, labels=labels, config=self.config, **loss_kwargs)

        else:
            raise ValueError(f"{self.config.loss_type} is not implemented or missing config arguments")

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
