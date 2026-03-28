import re
import string
import numpy as np
import logging

from typing import List, Dict
from .generation_metric import GenerationMetric
from .utils import is_refusal, get_tokens, normalize_text
import math

log = logging.getLogger("lm_polygraph")


class NumberMatch(GenerationMetric):
    """
    Calculates accuracy between model-generated texts and ground-truth.
    Two texts are considered equal if model-generated string in ground-truth.
    Basically, it's a simplified and relaxed accuracy metric.
    """
    _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

    def __init__(
        self,
        skips_refusals,
        epsilon=1e-2,
    ):
        super().__init__(["greedy_texts"], "sequence")
        self.normalize = True
        self.skips_refusals = skips_refusals
        self.epsilon = epsilon

    @staticmethod
    def _extract_number(text: str) -> float | None:
        """Extract the last number from text.

        Strips currency symbols, commas, and percent signs before matching.
        Returns None if no number is found.
        """
        cleaned = text.replace(",", "").replace("%", "").replace("$", "").strip()
        matches = NumberMatch._NUMBER_RE.findall(cleaned)
        if not matches:
            return None
        return float(matches[-1])

    def _match_single(self, prediction: str, reference: str) -> float:
        """Check if predicted number matches reference.

        Follows the original T2-RagBench / encourage implementation:
        https://github.com/uhh-hcds/encourage/blob/main/src/encourage/metrics/number_match.py
        """
        pred_num = self._extract_number(prediction)
        gold_num = self._extract_number(reference)

        if pred_num is None or gold_num is None:
            return 0.0

        a_star = abs(pred_num)
        a = abs(gold_num)

        if a < self.epsilon:
            return 1.0 if a_star < self.epsilon else 0.0
        if a_star < self.epsilon:
            return 0.0

        ratio = a_star / a
        try:
            q = ratio * 10 ** (-math.floor(math.log10(ratio)))
        except (ValueError, OverflowError):
            return 0.0
        if q > 5:
            q *= 0.1
        return 1.0 if abs(q - 1) < self.epsilon else 0.0

    def __str__(self):
        return f"NumberMatch_{self.skips_refusals}"

    def _score_single(self, output: str, target: str) -> int:
        if self.skips_refusals:
            if is_refusal(output):
                return float("nan")
        return self._match_single(output, target)

    def _normalize_text(self, text: str) -> str:
        text = normalize_text(text)

        return text

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates accuracy between stats['greedy_texts'] and target_texts.

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
