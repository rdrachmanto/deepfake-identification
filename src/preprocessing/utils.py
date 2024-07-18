from collections import Counter
import random
import re


def select_eligible_frames(num_frames: int, cut_amount: float) -> list[int]:
    if cut_amount >= 0.5:
        raise Exception("Cannot go 0.5 or above")

    lst = [x for x in range(num_frames)]
    remove_count = int(num_frames * cut_amount)
    return lst[remove_count:-(remove_count)]


def sample_frames_from_list(
    eligible_frames: list[int], n_frame: int, seed: int
) -> list[int]:
    random.seed(seed)
    return random.sample(eligible_frames, n_frame)


class IncompleteGatherer:
    def __init__(self, logfile: str) -> None:
        self.logfile = logfile

    def _read_from_log(self, preprocessor: str, dataset: str, dataclass: str):
        files: list[str] = []
        with open(self.logfile) as f:
            lines = f.read().splitlines()
            pattern = re.compile(rf".*{preprocessor}.*{dataset}.*{dataclass}.*")
            for l in lines:
                m = pattern.match(l)
                if m is not None:
                    files.append(m.group().split()[-3])
        return files

    def gather(self, preprocessor: str, dataset: str, dataclass: str):
        return Counter(self._read_from_log(preprocessor, dataset, dataclass))
