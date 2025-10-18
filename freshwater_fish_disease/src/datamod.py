from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class AlbumentationsImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)
        self.aug = transform
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = cv2.imread(path)[:, :, ::-1]  # BGR->RGB
        if self.aug: img = self.aug(image=img)["image"]
        return img, target

def get_augs(size):
    train_aug = A.Compose([
        A.LongestMaxSize(size), A.PadIfNeeded(size, size),
        A.RandomResizedCrop(size, size, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.7),
        A.HorizontalFlip(p=0.5), A.ColorJitter(0.1,0.1,0.1,0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()
    ])
    val_aug = A.Compose([
        A.LongestMaxSize(size), A.PadIfNeeded(size, size),
        A.CenterCrop(size, size),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()
    ])
    return train_aug, val_aug

def build_loaders(root, img_size, batch_size, num_workers):
    train_aug, val_aug = get_augs(img_size)
    paths = {k: Path(root)/k for k in ["train","val","test"]}

    ds_train = AlbumentationsImageFolder(paths["train"], transform=train_aug)
    ds_val   = AlbumentationsImageFolder(paths["val"],   transform=val_aug)
    ds_test  = AlbumentationsImageFolder(paths["test"],  transform=val_aug)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return ds_train, ds_val, ds_test, loader_train, loader_val, loader_test
