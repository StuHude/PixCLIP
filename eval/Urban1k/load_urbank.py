import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
xyc_test=[0.48145466]
xyc_test1=[0.26862954]
class UrbankDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "image")
        self.caption_dir = os.path.join(root_dir, "caption")
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            #transforms.Normalize(0.5, 0.26),
        ])

        # 获取所有图像文件名（不含扩展名）
        self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        self.image_ids.sort()  # 按数字顺序排序

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 图像路径
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # 掩码：全白
        dummy_mask = np.ones((224, 224), dtype=np.uint8) * 255
        mask_tensor = self.mask_transform(dummy_mask)

        # 文本路径
        caption_path = os.path.join(self.caption_dir, f"{image_id}.txt")
        with open(caption_path, "r", encoding="utf-8") as f:
            caption_value = f.read().strip()

        return img_tensor, mask_tensor, caption_value


if __name__=="__main__":
    root_dir="data/Urban1k"
    dataset=UrbankDataset(root_dir)
    dataloader=DataLoader(dataset, batch_size=4, shuffle=True)
    for img,mask,caption in dataloader:
        print(f'image size: {img.shape}')
        print(f'mask size: {mask.shape}')
        print(f'caption: {caption}')
        break
    
