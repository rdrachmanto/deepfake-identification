import os
import logging

from facenet_pytorch import MTCNN  # type: ignore[reportMissingTypeStubs]
import cv2

from src.preprocessing.extractors.detectors import MTCNNDetector
import src.preprocessing.extractors.utils as utils
import src.utils as coreutils


class MTCNNSampleExtractor(MTCNNDetector):
    def __init__(self, dataset_path: str, classes: list[str]) -> None:
        self.dataset_path = dataset_path
        self.classes = classes
        super().__init__()

    def extract(
        self, save_to: str, n_frame: int, cut_amount: float, seed: int, batch_size: int
    ):
        coreutils.create_directory(save_to, self.classes)
        for c in self.classes:
            videos = os.listdir(f"{self.dataset_path}/{c}")
            for b_loop in range(0, len(videos), batch_size):
                current_batch = videos[b_loop : b_loop + batch_size]
                for i, f in enumerate(current_batch):
                    capture = cv2.VideoCapture(f"{self.dataset_path}/{c}/{f}")
                    eligible_frames = utils.select_eligible_frames(
                        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                        cut_amount,
                    )

                    # Process the sampled frames
                    for i in utils.sample_frames_from_list(
                        eligible_frames, n_frame, seed
                    ):
                        _ = capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                        _, frame = capture.read()

                        pil_image, face_boxes = self.detect_face(frame=frame)  # type: ignore[reportUnknownVariableType]
                        if len(face_boxes) == 0:  # type: ignore[reportUnknownArgumentType]
                            logging.error(
                                f"MTCNN no face detected on {self.dataset_path}/{c}/{f} frame {i}"
                            )
                            continue

                        face = self.crop_frame_to_face(pil_image, face_boxes[0])  # type: ignore[reportUnknownArgumentType]
                        face.save(f"{save_to}/{c}/{f}_frame_{i}.jpg")

                    capture.release()


class MTCNNSeqExtractor(MTCNNDetector):
    def __init__(self, video_counts: dict[str, int]) -> None:
        self.video_counts = video_counts
        super().__init__()

    def extract(self, save_to: str, cut_amount: float):
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

                pil_image, face_boxes = self.detect_face(frame=frame)  # type: ignore[reportUnknownVariableType]
                if len(face_boxes) == 0:  # type: ignore[reportUnknownArgumentType]
                    continue

                face = self.crop_frame_to_face(pil_image, face_boxes[0])  # type: ignore[reportUnknownArgumentType]

                c = vf.split("/")[-2]
                f = vf.split("/")[-1]
                save_path = f"{save_to}/{c}/{f}_frame_{i}.jpg" 
                if os.path.exists(save_path):
                    continue

                face.save(save_path)
                recovered_frames += 1
            capture.release()
