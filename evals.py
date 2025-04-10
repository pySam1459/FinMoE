import torch
import pandas as pd
from functools import partial
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Dataset, load_dataset
from typing import Optional, Any

from FinMoE import FinMoE
from utils import DatasetArgs


def __preprocess_eval_a(tokenizer: PreTrainedTokenizer,
                        max_length: int,
                        example: dict[str, Any]):
    """Preprocessing function for FPB and Headline"""
    ## FPB and Headline are loaded as few-shot, convert to zeroshot for eval dataset
    zeroshot = example['input'].rsplit("\n\n", maxsplit=1)[-1]
    return tokenizer(zeroshot,
                     truncation=True,
                     padding="max_length",
                     max_length=max_length,
                     return_tensors="pt")

def __preprocess_eval_b(tokenizer: PreTrainedTokenizer,
                        max_length: int,
                        prompt_templates: dict[str, str],
                        example: dict[str, Any]):
    """Preprocessing function for Topics"""
    zeroshot = prompt_templates["Topics"].format(example["text"])
    return tokenizer(zeroshot,
                     truncation=True,
                     padding="max_length",
                     max_length=max_length,
                     return_tensors="pt")

def __eval_b_add_options(topic_options: list[str], example: dict[str, Any]):
    example["options"] = topic_options
    return example


def load_eval_dataset(tokenizer: PreTrainedTokenizer,
                      dataset_id: str,
                      args: DatasetArgs) -> Dataset:
    """
    Loads a datasets.Dataset containing the evaluation testset for each dataset used by FinMoE
    Possible datasets include (FPB, Headline, Topics)

    Args:
        tokenizer (PreTrainedTokenizer): model tokenizer from transformers library
        dataset_id (str): dataset identifier of the evaluation dataset to load
        args (DatasetArgs): arguments for dataset loading, see utils.py

    Returns:
        datasets.Dataset: evaluation dataset
    """

    if dataset_id in ["FPB", "Headline"]:
        print(f"Loading {dataset_id} dataset from AdaptLLM/finance-tasks")
        preprocess_partial = partial(__preprocess_eval_a, tokenizer, args.max_length)
        return load_dataset("AdaptLLM/finance-tasks", dataset_id, split="test").map(preprocess_partial, batched=False)
    
    elif dataset_id in ["Topics"]:
        dataset_path = args.paths[dataset_id]
        print(f"Loading {dataset_id} from path {dataset_path}")

        # topic options are "0", "1", "2", ..., "19"
        topic_options = [str(i) for i in range(len(args.topics))]
        add_opts_partial = partial(__eval_b_add_options, topic_options)

        testset_df = pd.read_csv(dataset_path / "test.csv",
                                 delimiter=args.del_mapping[dataset_id],
                                 names=args.names_mapping[dataset_id])
        
        preprocess_partial = partial(__preprocess_eval_b, tokenizer, args.max_length, args.prompt_templates)
        return (Dataset
                .from_pandas(testset_df)
                .map(preprocess_partial, batched=False)
                .map(add_opts_partial, batched=False)
                .rename_column("label", "gold_index")) ## align with "FPB" and "Headline"
    else:
        raise ValueError("Invalid dataset id. Only FPB, Headline, Topics")


def evaluate(model: PreTrainedModel,
             tokenizer: PreTrainedTokenizer,
             testset: Dataset,
             guidance = True,
             token_opts: Optional[list[int]] = None) -> dict[str, float]:
    """
    Computes metrics for a model on the testset provided.
    A token list can be provided to mask the model's possible outputs

    Args:
        model (PreTrainedModel): pytorch model being evaluated
        tokenizer (PreTrainedTokenizer): model tokenizer from transformers library
        testset (Dataset): test examples in a datasets.Dataset
        guidance (bool): if True, only tokens from `token_opts` argument will be considered with other tokens masked to zero
        token_opts (list[int] | None): a list of token values

    Returns:
        dict[str, float]: a mapping of metric name to metric value computed
    """

    if guidance and token_opts is None:
        raise ValueError("Guidance requires token options")

    correct = 0
    prog_bar = tqdm(testset)
    for i, example in enumerate(prog_bar):
        input_ids = torch.tensor(example["input_ids"])
        attn_mask = torch.tensor(example["attention_mask"])
        gen_idx = attn_mask.sum(dim=1).long() - 1  ## index at which the new token will be generated

        out = model.forward(input_ids=input_ids.to(model.device),
                            attention_mask=attn_mask.to(model.device))
        if guidance:
            gen_logits = out.logits[0, gen_idx, token_opts].cpu()
            local_argmax = torch.argmax(gen_logits, dim=-1).item()
            gen_token = token_opts[local_argmax]
        else:
            gen_logits = out.logits[0, gen_idx, :].cpu()
            gen_token = torch.argmax(gen_logits, dim=-1)

        gen_raw = tokenizer.decode(gen_token).strip(" ")
        if example["options"][example["gold_index"]] == gen_raw:
            correct += 1

        prog_bar.set_description(f"{100 * correct / (i+1):.2f}")
    
    return {
        "accuracy": correct / len(testset)
    }


def evaluate_FinMoE(model: FinMoE,
                    tokenizer: PreTrainedTokenizer,
                    testset: Dataset,
                    token_opts: list[int])  -> dict[str, float]:
    """
    Modified evaluate function to minimise the switching of adapters, speeds to eval times
      by pre-computed the routes for each sample in the dataset and grouping samples by expert route
    """
    gating_progbar = tqdm(testset, desc="Routing experts")
    gating_routes = torch.empty(len(testset), dtype=torch.long, device=model.device)
    for i, example in enumerate(gating_progbar):
        input_ids = torch.tensor(example["input_ids"]).to(model.device)
        attn_mask = torch.tensor(example["attention_mask"]).to(model.device)

        gate_scores = model.gate.forward(input_ids=input_ids, attention_mask=attn_mask)
        gating_routes[i] = torch.argmax(gate_scores, dim=-1).item()

    ## Once routes have been computed, group dataset samples
    indices = {expert_idx: [] for expert_idx in range(model.n_experts)}
    for i, expert_idx in enumerate(gating_routes.cpu().tolist()):
        indices[expert_idx].append(i)

    ## Split the dataset into n_expert subsets
    subsets = {expert_idx: testset.select(indices)
               for expert_idx, indices in indices.items()}

    ## Evaluation starts here
    correct = 0
    run_total = 0
    total = len(testset)
    prog_bar = tqdm(total=total)
    for expert_idx, subset in subsets.items():
        model.expert.set_adapter(str(expert_idx))

        for example in subset:
            input_ids = torch.tensor(example["input_ids"])
            attn_mask = torch.tensor(example["attention_mask"])
            gen_idx = attn_mask.sum(dim=1).long() - 1

            out = model.expert.forward(input_ids=input_ids.to(model.device),
                                       attention_mask=attn_mask.to(model.device))
            
            batch_indices = torch.arange(out.logits.size(0), device=out.logits.device)
            gen_subset = out.logits[batch_indices, gen_idx, token_opts].cpu()
            local_argmax = torch.argmax(gen_subset, dim=-1).item()
            gen_token = token_opts[local_argmax]

            gen_raw = tokenizer.decode(gen_token).strip(" ")
            if example["options"][example["gold_index"]] == gen_raw:
                correct += 1
                
            run_total += 1
            prog_bar.set_description(f"Evaluating | {100 * correct / run_total:.2f}")
            prog_bar.update(1)

    return {
        "accuracy": correct / total,
        "n_correct": correct,
        "n_total": total,
    }
