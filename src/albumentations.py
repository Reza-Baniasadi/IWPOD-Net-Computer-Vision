# data/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OCRDataset(Dataset):
    def __init__(self, root_dir, alphabets, transform=None):
        self.root_dir = root_dir
        self.alphabets = alphabets
        self.transform = transform
        self.img_paths = []
        self.labels = []
        for fname in os.listdir(root_dir):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                self.img_paths.append(os.path.join(root_dir, fname))
                self.labels.append(os.path.splitext(fname)[0])
        self.char_to_idx = {c: i+1 for i, c in enumerate(alphabets)}  # 0 = blank

    def __len__(self):
        return len(self.img_paths)

    def encode(self, text):
        return [self.char_to_idx[c] for c in text]

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        label = self.encode(self.labels[idx])
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, labels

def get_transforms(img_h, img_w):
    return A.Compose([
        A.Resize(img_h, img_w),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=5, p=0.5),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
