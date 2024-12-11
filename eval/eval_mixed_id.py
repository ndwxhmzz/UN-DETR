from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == "__main__":
    cocoGt = COCO('/data/datasets/Detection/coco/annotations/instances_val2017_mixed_ID.json')        
    cocoDt = cocoGt.loadRes('/data/lhm/DETR-test/train.json')  
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()