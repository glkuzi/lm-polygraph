import re
import string
import numpy as np
import logging

from typing import List, Dict
from .generation_metric import GenerationMetric
from .utils import is_refusal, get_tokens, normalize_text

log = logging.getLogger("lm_polygraph")


class RefusalMetric(GenerationMetric):
    """
    Calculates refusal accuracy between model-generated texts and ground-truth.
    """

    def __init__(
        self,
    ):
        super().__init__(["greedy_texts"], "sequence")
        self.normalize = True

    def __str__(self):
        return f"Refusal"

    def _score_single(self, output: str, target: str) -> int:
        if normalize_text(target) != "unknown":
            return float("nan")
        if is_refusal(output):
            return 1
        return 0

    def _normalize_text(self, text: str) -> str:
        text = normalize_text(text)

        return text

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates refusal accuracy between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
        Returns:
            np.ndarray: list of accuracies: 1 if generated text is equal to ground-truth and 0 otherwise.
        """
        greedy_texts = stats["greedy_texts"]

        result = []

        for hyp, ref in zip(greedy_texts, target_texts):
            result.append(self._score_single(hyp, ref))

        return np.array(result)
