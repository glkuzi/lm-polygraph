from .constants import TOP_K, SEED
from functools import partial
import random


def prepare_gpqa(
    dataset,
    input_column,
    output_column,
    n_shot,
    description,
    few_shot_prompt,
    prompt,    
):
    random.seed(SEED)


    x, y = [], []
    for inst in dataset:
        correct_answer = inst["Correct Answer"]
        incorrect_answers = [inst["Incorrect Answer 1"], inst["Incorrect Answer 2"], inst["Incorrect Answer 3"]]
        choices = [correct_answer] + incorrect_answers
        
        # Shuffle the answers randomly
        random.shuffle(choices)
        
        # Create a list of answer letters (A, B, C, D)
        answer_letters = ['A', 'B', 'C', 'D']
        
        # Find the index of the correct answer after shuffling
        correct_index = choices.index(correct_answer)
        
        # Get the corresponding letter
        correct_letter = answer_letters[correct_index]
        
        x.append(description + prompt.format(question=inst[input_column],
                                             choices=choices))
        
        y.append(correct_letter)

    return x, y


CONFIG = {
    "gpqa_diamond_instruct": {
        "name": ["Idavidrein/gpqa", "gpqa_diamond"],
        "train_split": "train",
        "test_split": "train",
        "prepare_func": partial(
            prepare_gpqa,
            input_column="Question",
            output_column="answer",
            n_shot=0,
            description="Given the following question and four candidate answers (A, B, C, and D), choose the best answer. Your response should contain only the selected option's letter (A, B, C, or D), not a complete sentence.\n\n",
            few_shot_prompt="\n\nQuestion:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:{answer}",
            prompt="Question:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:",
        ),
        "dataset": "gpqa_diamond_instruct",
    },
}
