#!/bin/bash
sh eval.sh
cd eval
python predict.py
python yolo_coco_ood.py
python eval_ood.py
