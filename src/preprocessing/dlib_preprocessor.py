from datetime import datetime
import os
import logging

import cv2
from cv2.typing import MatLike
import dlib  # type: ignore[reportMissingTypeStubs]

from src.preprocessing.base_preprocessor import BasePreprocessor
from src.utils import create_directory


class DlibPreprocessor(BasePreprocessor):
    def __init__(self, dataset_path: str, classes: list[str]) -> None:
        super().__init__(dataset_path, classes)

    def _detect_face(self, frame: MatLike):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        return detector(frame, 1)

    def _crop_frame_to_face(self, frame, bounding_box):
        # Normally, setting these to face.left() and the likes are enough
        # However, to handle if the faces are so close to the edge..
        # We need this to adjust the face borders to be within image boundary
        x, y, w, h = (
            max(0, bounding_box.left()),
            max(0, bounding_box.top()),
            min(frame.shape[1], bounding_box.width()),
            min(frame.shape[0], bounding_box.height())
        )

        return frame[y:y+h, x:x+w]

    def preprocess(self, save_to: str, n_frame: int, cut_amount: float, seed: int, batch_size: int):
        create_directory(save_to, self.classes)
        for c in self.classes:
            s_time = datetime.now()
            videos = os.listdir(f"{self.dataset_path}/{c}")
            for b_loop in range(0, len(videos), batch_size):
                current_batch = videos[b_loop:b_loop + batch_size]
                for i, f in enumerate(current_batch):
                    capture = cv2.VideoCapture(f"{self.dataset_path}/{c}/{f}")
                    eligible_frames = self._select_eligible_frames(
                        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                        cut_amount
                    )

                    # Process the sampled frames
                    for i in self._sample_frames_from_list(eligible_frames, n_frame, seed):
                        _ = capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                        _, frame = capture.read()

                        faces = self._detect_face(frame)
                        if not len(faces) > 0:
                            logging.error(f"Dlib no face detected on {self.dataset_path}/{c}/{f} frame {i}")
                            continue
                        
                        face_frame = self._crop_frame_to_face(frame, faces[0])
                        _ = cv2.imwrite(f"{save_to}/{c}/{f}_frame_{i}.jpg", face_frame)
                    capture.release()

            e_time = datetime.now()
            print(f"Extracting {self.dataset_path}/{c} to {save_to}/{c} done in {round((e_time - s_time).total_seconds(), 2)}s")
