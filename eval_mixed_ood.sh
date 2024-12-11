#!/bin/bash
sh eval.sh
cd eval
python predict_mixed.py
python yolo_coco_mixed_ood.py
python eval_mixed.py
