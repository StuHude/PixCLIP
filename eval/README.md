# Evaluation Guide

This folder contains evaluation scripts for multiple datasets. All commands assume you run from the repo root (`/data/cy/xiaoclip`).

## COCO Masked Classification
Script: `eval/COCO/eval_coco.py`
```bash
python eval/COCO/eval_coco.py \
  --ckpt /data/cy/xiaoclip/train/log/grit_1m/ablation_study_ROI/ckpt/iter_8700.pth \
  --ann-file /data/xyc/coco/annotations/instances_val2017.json \
  --image-root /data/xyc/coco/val2017 \
  --model EVA02-CLIP-B-16 \
  --batch-size 128
```

## DOCCI Retrieval
Script: `eval/docci/eval_docci.py`
```bash
python eval/docci/eval_docci.py \
  --ckpt /data/cy/xiaoclip/train/log/grit_1m/ablation_study_ROI/ckpt/iter_8700.pth \
  --root-dir eval/docci/docci \
  --clip_type EVA02-CLIP-B-16
```

## Flickr30k Retrieval
Script: `eval/Flickr30k/eval_flickr.py`
```bash
python eval/Flickr30k/eval_flickr.py \
  --ckpt /data/cy/xiaoclip/train/log/grit_1m/ablation_study_ROI/ckpt/iter_8700.pth \
  --root-dir eval/Flickr30k/data \
  --json-file eval/Flickr30k/data/test_caption.json \
  --clip_type EVA02-CLIP-B-16
```

## Urban1k Retrieval
Script: `eval/Urban1k/eval_urban.py`
```bash
python eval/Urban1k/eval_urban.py \
  --ckpt /data/cy/xiaoclip/train/log/grit_1m/ablation_study_ROI/ckpt/iter_8700.pth \
  --root-dir eval/Urban1k/data/Urban1k \
  --clip_type EVA02-CLIP-B-16
```

## RefCOCO Zero-Shot (rec_zs_test)
Script: `eval/rec_zs_test/main.py`
```bash
python eval/rec_zs_test/main.py \
  --input_file eval/rec_zs_test/reclip_data/refcoco_val.jsonl \
  --image_root eval/rec_zs_test/data/train2014 \
  --method parse \
  --box_representation_method full,blur \
  --box_method_aggregator sum \
  --clip_model cp0.0 \
  --clip_type aclip \
  --xiaoclip_ckpt /data/cy/xiaoclip/train/log/grit_1m/ablation_study_ROI/ckpt/iter_8700.pth \
  --cache_path eval/rec_zs_test/cache
```
See `eval/rec_zs_test/README.md` for full data prep.
