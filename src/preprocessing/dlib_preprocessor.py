from datetime import datetime
import os
import logging

import cv2
import dlib  # type: ignore[reportMissingTypeStubs]

from src.preprocessing.detectors import DlibDetector
import src.preprocessing.utils as utils
import src.utils as coreutils


class DlibSamplePreprocessor(DlibDetector):
    def __init__(self, dataset_path: str, classes: list[str]) -> None:
        self.dataset_path = dataset_path
        self.classes = classes
        super().__init__()

    def preprocess(self, save_to: str, n_frame: int, cut_amount: float, seed: int, batch_size: int):
        coreutils.create_directory(save_to, self.classes)
        for c in self.classes:
            s_time = datetime.now()
            videos = os.listdir(f"{self.dataset_path}/{c}")
            for b_loop in range(0, len(videos), batch_size):
                current_batch = videos[b_loop:b_loop + batch_size]
                for i, f in enumerate(current_batch):
                    capture = cv2.VideoCapture(f"{self.dataset_path}/{c}/{f}")
                    eligible_frames = utils.select_eligible_frames(
                        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                        cut_amount
                    )

                    # Process the sampled frames
                    for i in utils.sample_frames_from_list(eligible_frames, n_frame, seed):
                        _ = capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                        _, frame = capture.read()

                        faces = self.detect_face(frame)
                        if not len(faces) > 0:
                            logging.error(f"Dlib no face detected on {self.dataset_path}/{c}/{f} frame {i}")
                            continue
                        
                        face_frame = self.crop_frame_to_face(frame, faces[0])
                        _ = cv2.imwrite(f"{save_to}/{c}/{f}_frame_{i}.jpg", face_frame)
                    capture.release()

            e_time = datetime.now()
            print(f"Extracting {self.dataset_path}/{c} to {save_to}/{c} done in {round((e_time - s_time).total_seconds(), 2)}s")


class DlibListPreprocessor(DlibDetector):
    def __init__(self, video_counts: dict[str, int]) -> None:
        self.video_counts = video_counts
        super().__init__()

    def preprocess(self, save_to: str, cut_amount: float):
        for vf in list(self.video_counts.keys()):
            capture = cv2.VideoCapture(vf)
            eligible_frames = utils.select_eligible_frames(
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                cut_amount,
            )

            recovered_frames = 0
            for i in eligible_frames:
                if recovered_frames >= self.video_counts[vf]:
                    break

                _ = capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = capture.read()

                faces = self.detect_face(frame)
                if not len(faces) > 0:  # type: ignore[reportUnknownArgumentType]
                    continue

                face_frame = self.crop_frame_to_face(frame, faces[0])

                c = vf.split("/")[-2]
                f = vf.split("/")[-1]
                save_path = f"{save_to}/{c}/{f}_frame_{i}.jpg" 
                if os.path.exists(save_path):
                    continue

                _ = cv2.imwrite(f"{save_to}/{c}/{f}_frame_{i}.jpg", face_frame)
                recovered_frames += 1
            capture.release()
