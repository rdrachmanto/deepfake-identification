import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
)
LEARNING_RATE = 0.0002
