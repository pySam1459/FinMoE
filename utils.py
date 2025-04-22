import pandas as pd
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from datasets import Dataset, interleave_datasets
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Callable, Optional, Any


@dataclass
class DatasetArgs:
    expert_order: list[str]

    prompt_templates: dict[str, str]
    prompt_args: dict[str, list[str]]

    paths: dict[str, Path]
    columns: dict[str, list[str]]
    del_mapping: dict[str, str | None]
    names_mapping: dict[str, list[str] | None]

    topics: list[str]
    id2labels: dict[str, dict[Any, str]]
    labels_list: list[str]

    token_opts: dict[str, list[int]]
    token_list: list[int]

    rng_seed: int = 42
    max_length: int = 256


def get_adaptllm_path(base_path: Path) -> Path:
    with open(base_path / "refs" / "main", "r") as f_in:
        snapshot_ref = f_in.readline()
    return base_path / "snapshots" / snapshot_ref


def get_dataset_args(tokenizer: PreTrainedTokenizer, hub_basepath: Path) -> DatasetArgs:
    """
    This function loads many arguments used by the finetuning scripts to load datasets, ...
    """

    ## load_dataset causes an error, load directly from cached snapshot files
    topics = ['Analyst Update', 'Fed | Central Banks', 'Company | Product News', 'Treasuries | Corporate Debt', 'Dividend', 'Earnings', 'Energy | Oil', 'Financials', 'Currencies', 'General News | Opinion', 'Gold | Metals | Materials', 'IPO', 'Legal | Regulation', 'M&A | Investments', 'Macro', 'Markets', 'Politics', 'Personnel Change', 'Stock Commentary', 'Stock Movement']
    topic_options = "\n".join([f"{i} - {t}" for i, t in enumerate(topics)])

    expert_order = ["FPB", "Headline", "Topics"]

    id2labels = dict[str, dict[Any, str]]({
        "FPB": {"neutral": " Neutral", "positive": " Positive", "negative": " Negative"},
        "Headline": {0: " No", 1: " Yes"},
        "Topics": {i: str(i) for i in range(20)},
    })

    ## used to ensure order of token_list
    id2labels_ordered = {
        did: list(labels.values()) for did, labels in id2labels.items()
    }
    labels_list = [label for did in expert_order for label in id2labels_ordered[did]]

    token_opts = {
        did: [tokenizer.encode(v, add_special_tokens=False)[0] for v in id2labels_ordered[did]]
        for did in expert_order
    }
    token_list = [token for did in expert_order for token in token_opts[did]]
    

    return DatasetArgs(
        expert_order = expert_order,

        prompt_templates = {
            "FPB": "{0}\nQuestion: what is the sentiment?\nOptions:\n- Positive\n- Negative\n- Neutral",
            "Headline": "Headline: \"{0}\" Now answer this question: {1}",
            "Topics": "{0}\nNow classify the topic\nOptions 0-19:\n" + f"{topic_options} ",
        },
        prompt_args = {
            "FPB": ["text"],
            "Headline": ["text", "question"],
            "Topics": ["text"],
        },

        paths = {
            "FPB": get_adaptllm_path(hub_basepath / "datasets--AdaptLLM--FPB"),
            "Headline": get_adaptllm_path(hub_basepath / "datasets--AdaptLLM--Headline"),
            "Topics": hub_basepath / "datasets--Sujet--TopicClassification"
        },
        columns = {
            "FPB": ["text", "label"],
            "Headline": ["idx", "text", "question", "label", "subidx"],
            "Topics": ["label", "text"]
        },
        del_mapping = {
            "FPB": "\t",
            "Headline": "\t",
            "Topics": None ## regular comma-delimiter'd csv
        },
        names_mapping = {
            "FPB": None,
            "Headline": ["idx", "text", "question", "label", "subidx"],
            "Topics": ["label", "text"]
        },

        topics = topics,

        id2labels = id2labels,
        labels_list = labels_list,

        token_opts = token_opts,
        token_list = token_list,
    )


def load_train_datasets(args: DatasetArgs,
                        preprocess_func: Callable,
                        nrows: Optional[list[int]] = None):

    dataset_list = []
    for i, (dataset_id, dataset_path) in enumerate(args.paths.items()):
        train_subset = pd.read_csv(dataset_path / "train.csv",
                                    delimiter=args.del_mapping[dataset_id],
                                    names=args.names_mapping[dataset_id],
                                    nrows=nrows[i] if nrows is not None else None)

        __preprocess_func = partial(preprocess_func, args, dataset_id)
        dataset_list.append(Dataset
                            .from_pandas(train_subset)
                            .map(__preprocess_func,
                                batched=False,
                                remove_columns=args.columns[dataset_id])
                            .filter(lambda sample: len(sample["input_ids"]) <= args.max_length))

    n_datasets = len(dataset_list)
    return interleave_datasets(dataset_list, 
                               probabilities=[1/n_datasets]*n_datasets,
                               seed=args.rng_seed)
