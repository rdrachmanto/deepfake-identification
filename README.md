## Deepfake Identification

Identify deepfakes through original and modified `XceptionNet` models. Feed face images to the network and receive predictions whether the face is real, or a result of deepfake techniques. Built upon `pytorch` and `opencv-python`.

## Getting Started

It is recommended to use `conda` for this project. Below are the script to replicate the project environment.

Steps:
1. Create conda environment. below, the env is named `torchit`, this project is done on python 3.10.
2. Activate the conda env
3. Install packages through pip

```bash
conda create -n torchit python=3.10 -y
conda activate torchit
pip install -r requirements.txt
```
## Structure

All codes are available inside `src/`. Notebooks are available in `notebooks/`.
Scripts in the notebooks expects the data, both raw and preprocessed, to be inside the `data/raw` and `data/preprocessed` directory, respectively. Feel free to change accordingly. 

## License

This python script and its notebooks are licensed under MIT License.
