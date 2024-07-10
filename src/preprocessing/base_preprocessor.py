from abc import ABC, abstractmethod
import random


class BasePreprocessor(ABC):
    """Base preprocessor class"""

    def __init__(self, dataset_path: str, classes: list[str]) -> None:
        self.dataset_path = dataset_path
        self.classes = classes

    def _select_eligible_frames(self, num_frames: int, cut_amount: float) -> list[int]:
        if cut_amount >= 0.5:
            raise Exception("Cannot go 0.5 or above")
            
        lst = [x for x in range(num_frames)]
        remove_count = int(num_frames * cut_amount)
        return lst[remove_count:-(remove_count)]

    def _sample_frames_from_list(
        self, eligible_frames: list[int], n_frame: int, seed: int
    ) -> list[int]:
        random.seed(seed)
        return random.sample(eligible_frames, n_frame)
