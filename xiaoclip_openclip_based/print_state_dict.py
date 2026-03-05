
from transformers import AutoModel

# 加载预训练模型
model_name_or_path = "/data/cy/data/model/LLM2CLIP-Openai-L-14-224"
model = AutoModel.from_pretrained(model_name_or_path)

# 提取 state_dict
state_dict = model.state_dict()

# 查看部分参数名和形状
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
