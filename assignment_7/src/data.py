import kagglehub
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FashionProductDataset(Dataset):
    def __init__(self, records: pd.DataFrame, img_dir: Path, transform, label_col: str):
        self.records = records.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        img_path = self.img_dir / f"{int(row['id'])}.jpg"
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), int(row["label"])


def get_dataloaders(
    batch_size: int = 64,
    img_size: int = 64,
    label_col: str = "articleType",
    max_classes: int = 20,
    val_split: float = 0.2,
    seed: int = 42,
):
    dataset_path = Path(kagglehub.dataset_download("paramaggarwal/fashion-product-images-small"))

    csv_path = next(dataset_path.rglob("styles.csv"))
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    jpg_files = list(dataset_path.rglob("*.jpg"))
    img_dir = jpg_files[0].parent
    available_ids = {int(p.stem) for p in jpg_files if p.stem.isdigit()}

    df = df[df["id"].isin(available_ids)].copy()
    df = df.dropna(subset=[label_col])

    top_classes = df[label_col].value_counts().head(max_classes).index.tolist()
    df = df[df[label_col].isin(top_classes)].copy()

    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(top_classes))}
    df["label"] = df[label_col].map(class_to_idx)
    num_classes = len(class_to_idx)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df["label"]
    )

    import torch
    pin = torch.cuda.is_available()

    train_dataset = FashionProductDataset(train_df, img_dir, transform_train, label_col)
    val_dataset = FashionProductDataset(val_df, img_dir, transform_val, label_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    return train_loader, val_loader, num_classes, class_to_idx
