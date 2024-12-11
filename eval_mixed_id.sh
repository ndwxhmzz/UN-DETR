#!/bin/bash
sh eval.sh
cd eval
python predict_mixed_id.py
python yolo_coco_mixed_id.py
python eval_mixed_id.py
