import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 图像标准化参数
PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None):
        self.img_dir = os.path.join(root_dir, "flickr30k-images")  # 图像目录
        self.json_file = json_file  # 新的JSON标注路径
        
        # 图像变换
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ])

        # 掩码变换（全白）
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

        # 加载 JSON 数据
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 对每个条目提取 image_id（去除扩展名）
        self.image_entries = []
        for item in self.data:
            filename = item["filename"]
            image_id = os.path.splitext(filename)[0]
            self.image_entries.append({
                "image_id": image_id,
                "filename": filename,
                "captions": item["captions"]
            })

    def __len__(self):
        return len(self.image_entries)

    def __getitem__(self, idx):
        entry = self.image_entries[idx]
        image_id = entry["image_id"]

        # 加载图像
        img_path = os.path.join(self.img_dir, entry["filename"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # 全白掩码
        dummy_mask = np.ones((224, 224), dtype=np.uint8) * 255
        mask_tensor = self.mask_transform(dummy_mask)

        # 选择第一个 caption
        caption_value = entry["captions"][0]

        return img_tensor, mask_tensor, caption_value


if __name__ == "__main__":
    # 示例路径
    root_dir = "data"
    json_file = "data/test_caption.json"

    dataset = FlickrDataset(root_dir, json_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for img, mask, caption in dataloader:
        print(f'image size: {img.shape}')
        print(f'mask size: {mask.shape}')
        print(f'caption: {caption}')
        break
