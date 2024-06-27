## Deepfake Video Identification: Preprocessing
Sample preprocessing task for deepfake video identification, built on `opencv-python`. 

## Preprocessing Step
1. Load video to frames.
2. Identify heads or faces in the frame with `haarcascade`.
3. Save 10 frames with faces, next the images will be an input for training a CNN to classify. 

## License
MIT License
