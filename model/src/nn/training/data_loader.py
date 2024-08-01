from torchvision.datasets import ImageFolder  # type: ignore[reportMissingStubs]
from torchvision import transforms  # type: ignore[reportMissingStubs]
from torch.utils.data import DataLoader, random_split


def _split_dataset(dataset: ImageFolder, train_size: float, val_size: float):  # type: ignore[reportUnknownVariableType]
    train_count = int(train_size * len(dataset))
    val_count = int(val_size * len(dataset))
    test_count = len(dataset) - train_count - val_count

    return random_split(dataset, [train_count, val_count, test_count])  # type: ignore[reportUnknownVariableType]


def load_data(
    path: str,
    transforms: transforms.Compose,
    train_size: float,
    val_size: float,
    batch_size: int,
):  # type: ignore[reportUnknownVariableType]
    if not train_size + (2 * val_size) == 1.0:
        raise Exception("Splits must add up to 1.0")

    dataset = ImageFolder(root=path, transform=transforms)

    train_split, val_split, test_split = _split_dataset(dataset, train_size, val_size)  # type: ignore[reportUnknownVariableType]

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)  # type: ignore[reportUnknownVariableType]
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)  # type: ignore[reportUnknownVariableType]
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False)  # type: ignore[reportUnknownVariableType]

    return train_loader, val_loader, test_loader  # type: ignore[reportUnknownVariableType]
