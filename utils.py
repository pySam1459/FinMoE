from dataclasses import dataclass
from pathlib import Path
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Any


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

    max_length: int = 512


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
