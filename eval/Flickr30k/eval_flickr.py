import torch
from torch.utils.data import DataLoader, Subset
import sys
import argparse
from load_flickr import FlickrDataset
import os
from tqdm import tqdm
import gc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from xiaoclip import CustomCLIP, create_model, create_model_and_transforms

os.environ.setdefault("HF_HOME", "/data/xyc/cache")

def load_model(device, ckpt_path, clip_type):
    """
    加载预训练模型
    :param device: 设备类型，如 'cuda' 或 'cpu'
    :return: 加载好的模型和预处理函数
    """
    print("加载 XiaoClip 模型...")
    # TODO 使用xiaoclip的CustomCLIP
    model = create_model(
        model_name=clip_type, 
        force_custom_clip=True,
        pretrained=None,  # 不使用预训练权重
        use_alpha_channel=True,  # 启用alpha通道
        pre_extract_feature=False,
    )
    # model = model.float().cuda()
    # map_location = {'cuda:0': device}
    model=model.float().to(device)
    map_location = device
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt, strict=False)
    return model
    
    
def encode_all_texts_imgs(dataset, model, device, 
                            num_sample_retrieval=1000, 
                            batch_size=32,
                            num_workers=4,
                            shuffle=True,
                            use_amp=True):
    """
    对所有文本进行编码
    :param dataloader: 数据集加载器
    :param model: 编码器模型
    :param device: 设备
    :return: 所有文本的编码结果
    """
    # 创建子集
    subset = Subset(dataset, indices=range(min(num_sample_retrieval, len(dataset))))
    subset_loader = torch.utils.data.DataLoader(subset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, 
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                )


    all_text_features = []
    all_image_features = []
    all_captions = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(subset_loader, desc="Encoding all texts & imgs"):
            # 编码并移动到设备
            images = batch[0].to(device, non_blocking=True) # 异步传输 
            alpha = batch[1].to(device, non_blocking=True)
            captions = batch[2]
            # all_captions.append(captions[0])
            all_captions.extend(list(captions))


            # 显存优化：混合精度计算
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 编码文本
                text_features = model.encode_text(captions)
                all_text_features.append(text_features.cpu())

                # 编码图像
                # alpha = torch.ones((images.size(0), 1, images.size(2), images.size(3))).to(device)
                image_features = model.encode_image(images, alpha)
                all_image_features.append(image_features.cpu())

            del images, text_features, image_features, alpha
            gc.collect()
            torch.cuda.empty_cache()
    
    all_text_features = torch.cat(all_text_features, dim=0).to(device)
    all_image_features = torch.cat(all_image_features, dim=0).to(device) 

    return all_image_features, all_text_features, all_captions


def image_to_text_retrieval(
    all_image_features: torch.Tensor,
    all_text_features: torch.Tensor,
    all_captions: tuple[str],
    model: torch.nn.Module,
    device: torch.device,
    top_k: int = 1
) -> float:
    """
    优化版的图像到文本检索评估函数
    
    参数:
        all_image_features: 所有图像的编码特征 [N, d]
        all_text_features: 所有文本的编码特征 [N, d]
        all_captions: 文本描述列表 (用于调试和验证)
        model: 包含logit_scale的预训练模型
        device: 计算设备
        top_k: 计算top-k准确率 (默认1)
        
    返回:
        top-k检索准确率
    """
    # 数据验证
    assert len(all_image_features) == len(all_text_features) == len(all_captions), \
        f"数据长度不匹配: 图像({len(all_image_features)}) 文本({len(all_text_features)}) 描述({len(all_captions)})"

    # 确保特征在相同设备上
    image_features = all_image_features.to(device)
    text_features = all_text_features.to(device)
    
    # 计算所有图像-文本对的相似度矩阵 [N, N]
    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        sim_matrix = logit_scale * image_features @ text_features.t()  # [N, N]
    
    # 计算top-k准确率
    targets = torch.arange(len(text_features)).to(device)  # 假设1:1配对
    
    # 获取top-k预测
    _, topk_preds = sim_matrix.topk(top_k, dim=1)  # [N, k]
    correct = (topk_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
    
    accuracy = correct / len(image_features)
    
    return accuracy



def text_to_image_retrieval(
        all_image_features: torch.Tensor,
        all_text_features: torch.Tensor,
        all_captions: tuple[str],
        model: torch.nn.Module,
        device: torch.device,
        top_k: int = 1
    ) -> float:
    """
    优化版的文本到图像检索评估函数
    
    参数:
        all_image_features: 所有图像的编码特征 [N, d]
        all_text_features: 所有文本的编码特征 [N, d]
        all_captions: 文本描述列表 (仅用于调试)
        model: 包含logit_scale的预训练模型
        device: 计算设备
        top_k: 计算top-k准确率 (默认1)
        
    返回:
        top-k检索准确率
    """

    # 确保特征在相同设备上
    image_features = all_image_features.to(device)
    text_features = all_text_features.to(device)
    
    # 计算所有文本-图像对的相似度矩阵 [N, N]
    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        sim_matrix = logit_scale * text_features @ image_features.t()  # [N, N]
    
    # 计算top-k准确率
    correct = 0
    targets = torch.arange(len(image_features)).to(device)  # 假设1:1配对
    
    # 获取top-k预测
    _, topk_preds = sim_matrix.topk(top_k, dim=1)  # [N, k]
    correct = (topk_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
    
    accuracy = correct / len(image_features)
    
    return accuracy



def main(args):
    """
    主函数，作为程序的启动入口
    """
    print("加载数据")
    root_dir = args.root_dir
    json_file = args.json_file

    dataset = FlickrDataset(root_dir, json_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    ckpt_path=args.ckpt
    
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint {ckpt_path} not found, skipping...")
        return

    model = load_model(args.device, ckpt_path, args.clip_type)
    
    all_image_features, all_text_features, all_captions = encode_all_texts_imgs(dataset, model, 
                                                                                args.device, 
                                                                                args.num_sample_retrieval,
                                                                                args.batch_size,
                                                                                args.num_workers,
                                                                                args.shuffle)

    accuracy_i2t = image_to_text_retrieval(all_image_features, all_text_features, all_captions, model, args.device)
    # print(f"图片检索文本（I2T）的准确率: {accuracy_i2t * 100:.4f}%")
    
    accuracy_t2i = text_to_image_retrieval(all_image_features, all_text_features, all_captions, model, args.device)
    # print(f"文本检索图片（T2I）的准确率: {accuracy_t2i * 100:.1f}%")

    # # 存储结果
    # i2t_accuracies.append(accuracy_i2t)
    # t2i_accuracies.append(accuracy_t2i)
    
    print(f"I2T Acc={accuracy_i2t*100:.1f}%, T2I Acc={accuracy_t2i*100:.1f}%")
    
if __name__ == "__main__":

    parser =  argparse.ArgumentParser(description="Image-Text Retrieval Evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to use")
    parser.add_argument("--clip_type", type=str, default="EVA02-CLIP-B-16", help="The Type of XiaoClip model")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--root-dir", type=str, default="data", help="Flickr30k data directory")
    parser.add_argument("--json-file", type=str, default="data/test_caption.json", help="Flickr30k captions json")
    parser.add_argument("--num_sample_retrieval", type=int, default=1000, help="The number of samples to retrieve")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers")
    parser.add_argument("--shuffle", type=bool, default=False, help="Whether to shuffle the dataset")

    args = parser.parse_args()
    main(args)
