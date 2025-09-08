import datasets

from builders.base import CONFIG as base_config
from builders.babi_qa import CONFIG as babi_qa_config
from builders.coqa import CONFIG as coqa_config
from builders.mmlu import CONFIG as mmlu_config
from builders.person import CONFIG as person_config
from builders.trivia_qa import CONFIG as trivia_qa_config
from builders.wiki import CONFIG as wiki_config
from builders.wmt import CONFIG as wmt_config
from builders.truthfulqa import CONFIG as truthfulqa_config
from builders.samsum import CONFIG as samsum_config
from builders.xsum import CONFIG as xsum_config
from builders.gsm8k import CONFIG as gsm8k_config
from builders.nq import CONFIG as nq_config
from builders.gpqa import CONFIG as gpqa_config
from builders.sciq import CONFIG as sciq_config


DATASET_CONFIG = (
    base_config
    | babi_qa_config
    | coqa_config
    | mmlu_config
    | person_config
    | trivia_qa_config
    | wiki_config
    | wmt_config
    | truthfulqa_config
    | samsum_config
    | xsum_config
    | gsm8k_config
    | nq_config
    | gpqa_config
    | sciq_config
)


DATASETS_WITH_SUBSET = [
    "nq_instruct",
    "babi_qa",
    "sciq",
    "coqa_instruct",
    "triviaqa_instruct",
    "truthfulqa_instruct",
]

def build_dataset(dataset_name):
    config = DATASET_CONFIG[dataset_name]
    if isinstance(config["name"], list):
        dataset = datasets.load_dataset(
            *config["name"], trust_remote_code=True, num_proc=4
        )
    else:
        dataset = datasets.load_dataset(
            config["name"], trust_remote_code=True, num_proc=4
        )

    def prepare_dataset(split):
        if any([dataset_name in config["dataset"] for dataset_name in DATASETS_WITH_SUBSET]):
            x, y = config["prepare_func"](dataset=dataset[config[f"{split}_split"]], split=split, size=config[f"{split}_size"])
        else:
            x, y = config["prepare_func"](dataset=dataset[config[f"{split}_split"]])
        result_dataset = datasets.Dataset.from_dict({"input": x, "output": y})
        return result_dataset

    result = {}
    if "train_split" in config:
        result["train"] = prepare_dataset("train")
    if "test_split" in config:
        result["test"] = prepare_dataset("test")
    return datasets.DatasetDict(result)
