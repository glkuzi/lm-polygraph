from functools import partial


def prepare_sciq(
    dataset,
    split,
    size,
    input_column,
    output_column,
    description,
    prompt,    
):
    if size != -1 and size < len(dataset):
        dataset = dataset.select(range(size))
    x, y = [], []
    for inst in dataset:
        formatted_description = description.format(context=inst["support"])
        formatted_prompt = (
            formatted_description
            + prompt.format(
                question=inst[input_column],
            )
        )
        x.append(formatted_prompt)
        y.append(inst[output_column])
    return x, y


CONFIG = {
    "sciq_instruct": {
        "name": "sciq",
        "train_split": "train",
        "test_split": "validation",
        "train_size": 1000,
        "test_size": -1,
        "prepare_func": partial(
            prepare_sciq,
            input_column="question",
            output_column="correct_answer",
            description="Here's a short context:\n\n{context} (End of context)\n\nAnswer the following question as briefly as possible.",
            prompt="\n\nQuestion: {question}\nAnswer:",
        ),
        "dataset": "sciq_instruct",
    },
}  
