#!/bin/bash
sh eval.sh
cd eval
python predict_voc.py
python yolo_coco.py
python eval_voc.py
