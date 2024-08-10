from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def _split_dataset(dataset: ImageFolder, train_size: float, val_size: float):
    train_count = int(train_size * len(dataset))
    val_count = int(val_size * len(dataset))
    test_count = len(dataset) - train_count - val_count

    return random_split(dataset, [train_count, val_count, test_count])


def load_data(
    path: str,
    transforms: transforms.Compose,
    train_size: float,
    val_size: float,
    batch_size: int,
):
    if not train_size + (2 * val_size) == 1.0:
        raise Exception("Splits must add up to 1.0")

    dataset = ImageFolder(root=path, transform=transforms)

    train_split, val_split, test_split = _split_dataset(dataset, train_size, val_size)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
