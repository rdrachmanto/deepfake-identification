from datetime import datetime
import os
import logging

from cv2.typing import MatLike
from facenet_pytorch import MTCNN  # type: ignore[reportMissingTypeStubs]
from PIL import Image
import cv2

from src.preprocessing.base_preprocessor import BasePreprocessor
import src.utils as utils


class MTCNNPreprocessor(BasePreprocessor):
    def __init__(self, dataset_path: str, classes: list[str]) -> None:
        super().__init__(dataset_path, classes)

    def _detect_face(self, frame: MatLike):  # type: ignore[reportUnknownParameterType]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        mtcnn = MTCNN(keep_all=True)
        boxes, probs = mtcnn.detect(pil_image)  # type: ignore[reportAssignmentType]

        threshold = 0.9
        return pil_image, boxes[probs > threshold] if boxes is not None else []  # type: ignore[reportUnknownVariableType]

    def _crop_frame_to_face(self, frame: Image.Image, bounding_boxes: list[int]):
        boxes: list[int] = [int(coord) for coord in bounding_boxes]
        face = frame.crop((boxes[0], boxes[1], boxes[2], boxes[3]))
        return face

    def preprocess(self, save_to: str, n_frame: int, cut_amount: float, seed: int, batch_size: int):
        utils.create_directory(save_to, self.classes)
        for c in self.classes:
            s_time = datetime.now()
            videos = os.listdir(f"{self.dataset_path}/{c}")
            for b_loop in range(0, len(videos), batch_size):
                current_batch = videos[b_loop:b_loop + batch_size]
                for i, f in enumerate(current_batch):
                    capture = cv2.VideoCapture(f"{self.dataset_path}/{c}/{f}")
                    eligible_frames = self._select_eligible_frames(
                        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                        cut_amount,
                    )

                    # Process the sampled frames
                    for i in self._sample_frames_from_list(eligible_frames, n_frame, seed):
                        _ = capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                        _, frame = capture.read()

                        pil_image, face_boxes = self._detect_face(frame=frame)  # type: ignore[reportUnknownVariableType]
                        if len(face_boxes) == 0:  # type: ignore[reportUnknownArgumentType]
                            logging.error(f"MTCNN no face detected on {self.dataset_path}/{c}/{f} frame {i}")
                            continue

                        face = self._crop_frame_to_face(pil_image, face_boxes[0])  # type: ignore[reportUnknownArgumentType]
                        face.save(f"{save_to}/{c}/{f}_frame_{i}.jpg")
                    
                    capture.release()

            e_time = datetime.now()
            print(f"Extracting {self.dataset_path}/{c} to {save_to}/{c} done in {round((e_time - s_time).total_seconds(), 2)}s")
