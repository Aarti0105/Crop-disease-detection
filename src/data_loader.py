# Load and Prepare Data
# ============================================================


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LeafDataset(Dataset):
    """
    PyTorch DataLoader calls __len__ and __getitem__ to build batches.
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype("uint8")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(cfg):
    mean = cfg["IMG_MEAN"]
    std  = cfg["IMG_STD"]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


def prepare_data(images_path, labels_path, cfg):
    """
    Load data and return DataLoaders.

    Returns
    -------
    loaders      : dict — "train", "val", "test" DataLoaders
    test_labels  : integer labels for test set (for sklearn metrics)
    class_names  : list of 38 disease class names
    encoder      : LabelEncoder
    """

    images = np.load(images_path)
    labels = np.load(labels_path, allow_pickle=True)
    print(f"  {images.shape[0]:,} images  |  {len(np.unique(labels))} classes")

    # Convert string class names to integers
    encoder = LabelEncoder()
    labels_int = encoder.fit_transform(labels)
    class_names = list(encoder.classes_)

    unique, counts = np.unique(labels_int, return_counts=True)

    # 80% train+val, 20% test
    X_tv, X_test, y_tv, y_test = train_test_split(
        images, labels_int,
        test_size=cfg["TEST_SIZE"],
        random_state=cfg["SEED"],
        stratify=labels_int,
    )

    # 80% train, 20% val from remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=cfg["VAL_SIZE"],
        random_state=cfg["SEED"],
        stratify=y_tv,
    )

    print(f"  Train {X_train.shape[0]:,}  Val {X_val.shape[0]:,}  Test {X_test.shape[0]:,}")

    train_tf, val_tf = get_transforms(cfg)

    loaders = {
        "train" : DataLoader(
            LeafDataset(X_train, y_train, train_tf),
            batch_size=cfg["BATCH_SIZE"],
            shuffle=True,
            num_workers=cfg["NUM_WORKERS"],
        ),
        "val"   : DataLoader(
            LeafDataset(X_val, y_val, val_tf),
            batch_size=cfg["BATCH_SIZE"],
            shuffle=False,
            num_workers=cfg["NUM_WORKERS"],
        ),
        "test"  : DataLoader(
            LeafDataset(X_test, y_test, val_tf),
            batch_size=cfg["BATCH_SIZE"],
            shuffle=False,
            num_workers=cfg["NUM_WORKERS"],
        ),
    }
    return loaders, y_test, class_names, encoder