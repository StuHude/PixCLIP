
创建于2025.4.12，主要是因为自己发现自己代码加载得不对.

参考 https://huggingface.co/microsoft/LLM2CLIP-EVA02-B-16

可以发现，我并没有将LLM2CLIP-EVA02-B-16.pt加载进去，也就是说，之前的训练，基本等同于我重新训练了一个visual encoder

这里我将原本的XiaoClip改为了`xiaoclip-v1` 文件夹，并基于LLM2CLIP/evaclip再做进一步的开发




