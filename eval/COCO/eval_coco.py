import torch
import torch.distributed as dist
from tqdm import tqdm
from mask_image_test import COCO_Masked_Test
import os
from dataclasses import dataclass, field, asdict
import sys
import yaml
import os
import subprocess
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/xyc/cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from xiaoclip import CustomCLIP, create_model, create_model_and_transforms

alpha_clip_dir = REPO_ROOT / "eval" / "AlphaCLIP"
if alpha_clip_dir.exists():
    sys.path.insert(0, str(alpha_clip_dir))
import alpha_clip

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    
    # if dist is available and initialized, then return True
    return True

simple_templates = [
    'a photo of a {}.'
]

def zeroshot_classifier(classnames, templates, model, alphaclip, local_rank=0):
    """
    classnames: 类别名称列表
    templates: 文本提示模板列表( a photo of a {}.)
    model: CLIP模型对象
    alphaclip (bool): 是否使用alpha-CLIP的tokenizer
    local_rank (int, optional): 当前设备的本地rank。默认为0
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts=[template.format(classname) for template in templates]
            if alphaclip:
                texts = alpha_clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings.to(f'cuda:{local_rank}')
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # TODO: 如果是CustomCLIP是不是就不需要这一步
            # 计算所有模板嵌入的均值，得到该类别的代表性嵌入向量，并再次归一化
            class_embedding = class_embeddings.mean(dim=0)
            # 确保所有类别向量位于单位超球面上
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(f'cuda:{local_rank}')
        # 返回堆叠后的嵌入向量，shape为[embedding_dim, num_classes]
        return zeroshot_weights
    
class CLIP_Clean_test():
    def __init__(self, model='ViT-B/16', batch_size=None, alphaclip=False, local_rank=0, ann_file=None, image_root=None):
        
        if model == 'ViT-L/14@336px':
            self.hi_res = True
        else:
            self.hi_res = False
            
        self.batch_size = batch_size
        self.alpha_clip = alphaclip
        self.local_rank = local_rank
        self.ann_file = ann_file
        self.image_root = image_root
        
        # TODO 使用xiaoclip的CustomCLIP
        self.model = create_model(
            model_name=model, 
            force_custom_clip=True,
            pretrained=None,  # 不使用预训练权重
            use_alpha_channel=True,  # 启用alpha通道
            pre_extract_feature=False,
        )
        
        self.model = self.model.float().cuda()
        
    @torch.no_grad()
    def test_epoch(self, dataloader, rank=None):
        temp_corr_dict=dict()
        for i, (images, masks, target) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            target = target.cuda()
            masks = masks.cuda()
            # image_features = self.model.visual(images, masks)
            # image_features = self.model.visual(images)
            image_features = self.model.encode_image(images, masks)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = torch.matmul(image_features, self.text_embeddings)
            pred = score.topk(1, dim=1)[1].squeeze(dim=1)
            pred_5 = score.topk(5, dim=1)[1].squeeze(dim=1)
            for i in range(target.shape[0]):
                if target[i].item() not in temp_corr_dict:
                    temp_corr_dict[target[i].item()] = [0, 0, 0]
                temp_corr_dict[target[i].item()][0] += 1
                if target[i].item() == pred[i].item():
                    temp_corr_dict[target[i].item()][1] += 1
                if target[i].item() in pred_5[i].tolist():
                    temp_corr_dict[target[i].item()][2] += 1
        return temp_corr_dict
    
    def test(self, ckpt_path):
        if ckpt_path is not None:
            map_location = {'cuda:0': f'cuda:{self.local_rank}'}
            
            ckpt = torch.load(ckpt_path, map_location=map_location)
            self.model.load_state_dict(ckpt, strict=False)            

            # print(f"load resumed checkpoint: {ckpt_path}")
            logging.info(f"load resumed checkpoint: {ckpt_path}")
        
        self.model.eval()

        testset = COCO_Masked_Test(ann_file=self.ann_file or "/data/xyc/coco/annotations/instances_val2017.json",
                                   root_directory=self.image_root or "/data/xyc/coco/val2017",
                                   hi_res=self.hi_res)
        self.text_embeddings = zeroshot_classifier(testset.classes, simple_templates, self.model, self.alpha_clip, self.local_rank)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, num_workers=16, pin_memory=True)
        with torch.no_grad():
            temp_corr_dict = self.test_epoch(testloader)
            if is_dist_avail_and_initialized():
                output = [None] * dist.get_world_size()
                dist.all_gather_object(output, temp_corr_dict)
            else:
                output = [temp_corr_dict]
            # 在主进程中进行聚合
            if self.local_rank == 0:
                final_dict = dict()
                for dic in output:
                    for k, v in dic.items():
                        if k not in final_dict.keys():
                            final_dict[k] = v
                        else:
                            final_dict[k][0] += v[0]
                            final_dict[k][1] += v[1]
                            final_dict[k][2] += v[2]
                acc1 = 0.0
                acc5 = 0.0
                num_class = 0
                for v in final_dict.values():
                    acc1 += v[1] / v[0]
                    acc5 += v[2] / v[0]
                    num_class += 1
                acc1 = acc1 / num_class
                acc5 = acc5 / num_class
                # print("=====================================")
                # print(f"test mean of per class acc-1 step 0: {acc1}")
                # print(f"test mean of per class acc-5 step 0: {acc5}")
                # print("=====================================")
                logging.info("=====================================")
                logging.info(f"test mean of per class acc-1 step 0: {acc1}")
                logging.info(f"test mean of per class acc-5 step 0: {acc5}")
                logging.info("=====================================")
        return acc1, acc5
    
    def setup_distributed(backend="nccl", port=None):
        """Initialize distributed training environment.
        support both slurm and torch.distributed.launch
        see torch.distributed.init_process_group() for more details
        """
        num_gpus = torch.cuda.device_count()

        if "SLURM_JOB_ID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29991"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(rank % num_gpus)
            os.environ["RANK"] = str(rank)
        else:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(rank % num_gpus)

        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )
        return rank % num_gpus

# 添加StreamToLogger类
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if line.strip():
                self.logger.log(self.log_level, line.rstrip())
    
    def flush(self):
        pass    

def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


@dataclass
class Config:
    exp_name: str = "ablation_study_ROI_07241914"
    exp_root: str = "./cocoeval_exp"    
    model: str = "EVA02-CLIP-B-16"
    #ckpt_dir: str = "/data/cy/xiaoclip/train/log/grit_1m/rebuild_xiao_clip_visual_proj_1.7m_lr1e-4/ckpt"
    # ckpt_dir: str = "/data/cy/xiaoclip/train/log/grit_1m/shadow_0414_1.7m_visual_epoch15_lr1e-5/ckpt"
    ckpt_dir: str="/data/cy/xiaoclip/train/log/grit_1m/ablation_study_ROI/ckpt"
    batch_size: int = 128
    start_iter: int = 28000
    end_iter: int = 28000
    step: int = 1000
    ann_file: str = "/data/xyc/coco/annotations/instances_val2017.json"
    image_root: str = "/data/xyc/coco/val2017"


    @property
    def exp_dir(self):
        """返回完整实验路径"""
        os.makedirs(self.exp_root, exist_ok=True)
        return os.path.join(self.exp_root, self.exp_name)
    
    def setup_logging(self):
        """设置日志目录和文件"""
        # os.makedirs(self.log_dir, exist_ok=True)
        log_dir = os.path.join(self.exp_dir, "logs")
        log_file = os.path.join(log_dir, f"{self.exp_name}.log")
        # 获取根logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除已有handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 添加文件handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # 重定向标准输出
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

    
    def setup_dirs(self):
        """创建实验所需子目录"""
        os.makedirs(os.path.join(self.exp_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "results"), exist_ok=True) 
        os.makedirs(os.path.join(self.exp_dir, "logs"), exist_ok=True)




if __name__ == "__main__":

    import warnings
    import os
    import logging
    import argparse

    parser = argparse.ArgumentParser(description="COCO masked classification eval")
    parser.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path (optional).")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--start-iter", type=int, default=None, help="Start iter (inclusive).")
    parser.add_argument("--end-iter", type=int, default=None, help="End iter (inclusive).")
    parser.add_argument("--step", type=int, default=None, help="Step size.")
    parser.add_argument("--model", type=str, default=None, help="Model name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--exp-root", type=str, default=None, help="Experiment root.")
    parser.add_argument("--ann-file", type=str, default=None, help="COCO annotations file.")
    parser.add_argument("--image-root", type=str, default=None, help="COCO image root.")
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="Override CUDA_VISIBLE_DEVICES.")
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    warnings.filterwarnings("ignore")


    config = Config()
    if args.ckpt_dir:
        config.ckpt_dir = args.ckpt_dir
    if args.start_iter is not None:
        config.start_iter = args.start_iter
    if args.end_iter is not None:
        config.end_iter = args.end_iter
    if args.step is not None:
        config.step = args.step
    if args.model:
        config.model = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.exp_root:
        config.exp_root = args.exp_root
    if args.ann_file:
        config.ann_file = args.ann_file
    if args.image_root:
        config.image_root = args.image_root

    config.setup_dirs()
    config.setup_logging()
    # 保存配置文件
    os.makedirs(config.exp_dir, exist_ok=True)
    config_path = os.path.join(config.exp_dir, "configs", "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(asdict(config), f)
    
    ckpt_paths = []
    if args.ckpt:
        ckpt_paths = [args.ckpt]
    else:
        for i in range(config.start_iter, config.end_iter + 1, config.step):
            ckpt_paths.append(os.path.join(config.ckpt_dir, f"iter_{i}.pth"))

    import matplotlib.pyplot as plt

    acc1_list = []
    acc5_list = []
    ckpt_names = []
    local_rank = 0 #setup_distributed()
    
    test = CLIP_Clean_test(model=config.model,
                           batch_size=config.batch_size,
                           alphaclip=False,
                           local_rank=local_rank,
                           ann_file=config.ann_file,
                           image_root=config.image_root)

    for ckpt_path in ckpt_paths:
        logging.info(f"ckpt_path: {ckpt_path}")
        acc1, acc5 = test.test(ckpt_path if ckpt_path else None)
        print(f'acc1: {acc1}')
        print(f'acc5:{acc5}')
        # acc1_list.append(acc1)
        # acc5_list.append(acc5)
        # ckpt_names.append( f"{i}" if ckpt_path else "0")
        # plt.figure(figsize=(10, 5))
        # plt.plot(ckpt_names, acc1_list, label='Acc1', marker='o')
        # plt.plot(ckpt_names, acc5_list, label='Acc5', marker='o')
        # plt.xlabel('step')
        # plt.ylabel('Accuracy')
        # plt.title('Acc1 and Acc5 over different checkpoints')
        # plt.legend()
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # # 修改结果保存路径
        # plt.savefig(os.path.join(config.exp_dir, "results", f"{config.exp_name}.png"))        
