from abc import ABC

import cv2
from cv2.typing import MatLike
from PIL import Image
from facenet_pytorch import MTCNN  # type: ignore[reportMissingTypeStubs]


class MTCNNDetector(ABC):
    def detect_face(self, frame: MatLike):  # type: ignore[reportUnknownParameterType]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        mtcnn = MTCNN(keep_all=True)
        boxes, probs = mtcnn.detect(pil_image)  # type: ignore[reportAssignmentType]

        threshold = 0.9
        return pil_image, boxes[probs > threshold] if boxes is not None else []  # type: ignore[reportUnknownVariableType]

    def crop_frame_to_face(self, frame: Image.Image, bounding_boxes: list[int]):
        boxes: list[int] = [int(coord) for coord in bounding_boxes]
        face = frame.crop((boxes[0], boxes[1], boxes[2], boxes[3]))
        return face
