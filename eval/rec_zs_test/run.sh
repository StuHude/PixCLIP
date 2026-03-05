
python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model cp0.0 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache_1 --part "8,0"
