import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

class DOCCIDataset(Dataset):
    def __init__(self, root_dir, json_file="docci_descriptions.jsonlines", transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")

        with open(os.path.join(root_dir, json_file), "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item["image_file"]
        description = item["description"]

        # 加载图像
        img_path = os.path.join(self.image_dir, image_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # 掩码：全白
        dummy_mask = np.ones((224, 224), dtype=np.uint8) * 255
        mask_tensor = self.mask_transform(dummy_mask)

        return img_tensor, mask_tensor, description

if __name__ == "__main__":
    root_dir = "docci"
    dataset = DOCCIDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for img, mask, caption in dataloader:
        print(f'image size: {img.shape}')
        print(f'mask size: {mask.shape}')
        print(f'caption: {caption}')
        break
