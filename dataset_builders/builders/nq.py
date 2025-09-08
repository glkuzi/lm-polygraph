from .constants import TOP_K, SEED
from functools import partial
import copy
from datasets import load_dataset


def prepare_nq(
    dataset,
    split,
    size,
    input_column,
    output_column,
    few_shot_split,
    few_shot_dataset_func,
    n_shot,
    prompt,    
):
    import numpy as np

    np.random.seed(SEED)

    few_shot_dataset = few_shot_dataset_func()
    if n_shot > 0:
        if few_shot_split == split:
            indices = np.arange(len(few_shot_dataset))
            few_shot_indices = np.random.choice(indices, n_shot)
            indices = np.delete(indices, few_shot_indices)
            few_shot_dataset = copy.deepcopy(few_shot_dataset).select(few_shot_indices)
            dataset = dataset.select(indices)
        else:
            indices = np.arange(len(few_shot_dataset))
            few_shot_indices = np.random.choice(indices, n_shot)
            few_shot_dataset = copy.deepcopy(few_shot_dataset).select(few_shot_indices)

    if size != -1 and size < len(dataset):
        dataset = dataset.select(range(size))
    x, y = [], []
    if n_shot > 0:
        few_shot_ids = np.random.choice(
            len(few_shot_dataset), n_shot, replace=False
        )
        few_shot_data = few_shot_dataset.select(few_shot_ids)
        formatted_few_shot_prompt = ""
        for inst in few_shot_data:
            formatted_few_shot_prompt += (
                prompt.format(
                    question=inst[input_column].strip(),
                    answer=inst[output_column][0],
                )
                + "\n"
            )
    for inst in dataset:
        x.append(
            formatted_few_shot_prompt
            + prompt.format(
                question=inst[input_column],
                answer="",
            )
        )
        y.append([alias for alias in inst[output_column]])
    return x, y


CONFIG = {
    "nq_instruct": {
        "name": "nq_open",
        "train_split": "train",
        "test_split": "validation",
        "train_size": 1000,
        "test_size": 1000,
        "prepare_func": partial(
            prepare_nq,
            input_column="question",
            output_column="answer",
            few_shot_split="train",
            few_shot_dataset_func=partial(
                load_dataset,
                path="nq_open",
                split="train",
            ),
            n_shot=5,
            prompt="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nQuestion: {question}?\nAnswer:{answer}",
        ),
        "dataset": "nq_instruct",
    },
}
