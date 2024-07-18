## Deepfake Identification

Identify deepfakes through original and modified `XceptionNet` models. Feed face images to the network and receive predictions whether the face is original/real, or the result of deepfake techniques. Built upon `tensorflow` and `opencv-python`.

## Getting Started

It is recommended to use `conda` for this project. Below are the script to replicate the project environment.

Steps:
1. Create conda environment. below, the env is named `torchit`.
2. Activate the conda env
3. Install `dlib` through `conda-forge` channel. **DO NOT** use pip for `dlib` as it requires numerous cublas, cudnn and cuda-related libraries.
4. Install other packages through pip

```bash
conda create -n torchit python --yes
conda activate torchit
conda install -c conda-forge dlib --yes
pip install torch torchvision facenet-pytorch opencv-python 
```
Install and run `jupyterlab` (optionally) through pip as well with:

```bash
pip install jupyterlab jupyterlab-lsp  # Install jupyterlab and LSP (language server protocol) to enable documentation, and error checkings
jupyter-lab  # run jupyterlab
```

## Structure

All codes are available inside `src/`. Notebooks are available in `notebooks/`

## License

This python script and its notebooks are licensed under MIT License.
