import torch
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Dataset
from typing import Optional


def evaluate(model: PreTrainedModel,
             tokenizer: PreTrainedTokenizer,
             testset: Dataset,
             guidance = True,
             tok_opts: Optional[list[int]] = None) -> dict[str, float]:

    if guidance and tok_opts is None:
        raise ValueError("Guidance requires token options")

    correct = 0
    prog_bar = tqdm(testset)
    for i, example in enumerate(prog_bar):
        input_ids = torch.tensor(example["input_ids"])
        attn_mask = torch.tensor(example["attention_mask"])
        gen_idx = attn_mask.sum(dim=1).long() - 1

        out = model.forward(input_ids=input_ids.to(model.device),
                            attention_mask=attn_mask.to(model.device))
        logits = out.logits.cpu()
        
        gen_logits = logits[torch.arange(logits.size(0)), gen_idx, :] # (B, C)
        if guidance:
            subset = gen_logits[0, tok_opts]
            local_argmax = torch.argmax(subset).item()
            gen_tokens = tok_opts[local_argmax]
        else:
            gen_tokens = torch.argmax(gen_logits, dim=-1)

        gen_raw = tokenizer.decode(gen_tokens).strip(" ")
        if example["options"][example["gold_index"]] == gen_raw:
            correct += 1

        prog_bar.set_description(f"{100 * correct / (i+1):.2f}")
    
    return {
        "accuracy": correct / len(testset)
    }
