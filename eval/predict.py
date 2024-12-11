#coding=gb2312
import cv2
from PIL import Image
import numpy as np
import os
import time
 
import torch
from torch import nn
import torchvision.transforms as T
import sys
sys.path.append("..")
from main import get_args_parser as get_main_args_parser
from models import build_model
import numpy as np
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))
 
# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def dnms(bboxes, scores, threshold=0.6):
    x1 = torch.tensor(bboxes[:, 0])
    y1 = torch.tensor(bboxes[:, 1])
    x2 = torch.tensor(bboxes[:, 2])
    y2 = torch.tensor(bboxes[:, 3])
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    
    order = order.cpu()
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]

        center_x1 = (x1[i] + x2[i]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
        center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2

        diagonal_dist = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

        # 计算enclose框的左上角和右下角坐标
        enclose_x1 = torch.min(x1[i], x1[order[1:]])
        enclose_y1 = torch.min(y1[i], y1[order[1:]])
        enclose_x2 = torch.max(x2[i], x2[order[1:]])
        enclose_y2 = torch.max(y2[i], y2[order[1:]])

        # 计算enclose框的宽度和高度
        enclose_width = enclose_x2 - enclose_x1
        enclose_height = enclose_y2 - enclose_y1

        # 计算enclose框的对角线距离的平方
        enclose_diagonal_dist = enclose_width ** 2 + enclose_height ** 2

        # 计算DIOU
        diou = iou - (diagonal_dist / enclose_diagonal_dist)
        iou = torch.squeeze(diou)
        idx = np.nonzero(iou <= threshold)
        if idx.numel() == 0:
            break
           
        order = order[idx+1]
      
    return torch.LongTensor(keep)   

# 15pth 0.4nms 0.55 obj
def nms(bboxes, scores, threshold=0.4): # threshold the lower proposal the little
        x1 = torch.tensor(bboxes[:,0])
        y1 = torch.tensor(bboxes[:,1])
        x2 = torch.tensor(bboxes[:,2])
        y2 = torch.tensor(bboxes[:,3])
        areas = (x2-x1)*(y2-y1)   
        _, order = scores.sort(0, descending=True)   
        order = order.cpu() 
        keep = []
        while order.numel() > 0:      
            if order.numel() == 1:    
                i = order.item()
                keep.append(i)
                break
            else:
               
                i = order[0].item()   
                keep.append(i)

            
            xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

            iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
          
            iou = torch.squeeze(iou)
            idx = np.nonzero(iou <= threshold)
            if idx.numel() == 0:
                break
           
            order = order[idx+1]
      
        return torch.LongTensor(keep)   
    
    
def save_result(img_name, boxes, prob, obj, img_size, ori_prob):
    
    img_name = img_name.split('.')[0]
    
    output_dir = '/data/lhm/UN_DETR/pesudo_label/Voc_txt_result'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    with open(os.path.join(output_dir, img_name + '.txt'), 'a') as f:
    # img_size [h ,w, c]
        if len(boxes.shape) == 1:
            xmin, ymin, xmax, ymax = boxes
            width, height = (xmax - xmin), (ymax - ymin)
            x_center, y_center = (xmax - width / 2), (ymax - height / 2)

            scale_x, scale_y, scale_width, scale_height = (x_center / img_size[1]), (y_center / img_size[0]),\
                (width / img_size[1]), (height / img_size[0])


            f.write('81' + ' ' + str(scale_x) + ' ' + str(scale_y) + ' ' + str(scale_width) + ' ' + str(scale_height) + ' ' + str(obj[0].cpu().item()) +'\n')
        else:
            for index, coordinate in enumerate(boxes):
                
                xmin, ymin, xmax, ymax = coordinate
                if (xmax - xmin) / img_size[1] > 0.9 and (ymax - ymin) / img_size[0] > 0.9:
                    continue
                width, height = (xmax - xmin), (ymax - ymin)
                x_center, y_center = (xmax - width / 2), (ymax - height / 2)

                scale_x, scale_y, scale_width, scale_height = (x_center / img_size[1]), (y_center / img_size[0]),\
                    (width / img_size[1]), (height / img_size[0])
               
         
                # if np.max(ori_prob, 1)[index] < 0.7:
                f.write('81' + ' ' + str(scale_x) + ' ' + str(scale_y) + ' ' + str(scale_width) + ' ' + str(scale_height) + ' ' + str(obj[index].cpu().item()) + '\n')
                # else:
                #     print('now have id')
                #     print(prob)
                #     exit(0)
                #     f.write(str(np.argmax(prob.cpu().item(), 1)[index]) + ' ' + str(scale_x) + ' ' + str(scale_y) + ' ' + str(scale_width) + ' ' + str(scale_height) + ' ' + str(obj[index].cpu().item()) +'\n')
        f.close()

# plot box by opencv
def plot_result(pil_img, prob, boxes, obj, ori_prob,  save_name=None, imshow=False, imwrite=False, save_txt=False):
    prob = prob.unsqueeze(1)
    
    judge_score = obj 

    
    keep = nms(boxes, judge_score, 0.35)
    
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    LABEL =['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    img_size = opencvImage.shape
    
    prob = prob[keep]
    boxes = boxes[keep]
    obj = obj[keep]
    ori_prob = ori_prob[keep]

    # boxes = un_boxes
    if save_txt:
        save_result(save_name, boxes, prob, obj, img_size, ori_prob)    
        imshow = False
        imwrite = False
    if obj.shape[0] == 1:
        p = prob
        xmin, ymin, xmax, ymax = boxes
        object_score = obj
        cl = p.argmax()
        label_text = '{}%'.format(round(float(object_score.item()), 2))
 
        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2)
    else:
        for p, (xmin, ymin, xmax, ymax), object_score in zip(prob, boxes, obj):
            if (xmax - xmin) / opencvImage.shape[1] > 0.9 and (ymax - ymin) / opencvImage.shape[0] > 0.9:
                continue

            
            cl = p.argmax()
            label_text = '{}%'.format(round(float(object_score.item()), 2))
     
            cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
            cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2)
    
    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)
 
    if imwrite:
        if not os.path.exists("./result/ori_pred"):
            os.makedirs('./result/ori_pred')
        cv2.imwrite('./result/ori_pred/{}'.format(save_name), opencvImage)
# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
   
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
 
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b
 
def load_model(model_path , args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path) # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model
 
# 图像的推断
def detect(im, model, transform, prob_threshold=0.8):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].sigmoid()[0, :, :-1]
    probas = torch.sum(probas, dim=1)

    
    obj_cls = outputs['pred_logits'].sigmoid()[0, :, :-1]
    obj_cls = torch.sum(obj_cls, dim=1).unsqueeze(1)
    
    obj_score =  outputs['obj'][0]
  

    
    # keep_obj = torch.logical_and(outputs['obj'][0] > 0.7, obj_cls > 0.55)
    keep_obj = obj_score > 0.57

    # keep_obj2 = torch.unsqueeze(probas > 0.2, dim=1)

    
    ori_probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    ori_probas = ori_probas.cpu().detach().numpy()
  
    keep_obj = keep_obj
    keep_obj = torch.squeeze(keep_obj).cpu().detach().numpy()
    
  

    # objectness = outputs['obj'][0, keep_obj]
    objectness = obj_score[keep_obj]
    
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep_obj], im.size)


    end = time.time()
    

    return probas[keep_obj], bboxes_scaled, end - start, objectness, ori_probas[keep_obj]
 
 
if __name__ == "__main__":
    
    
    main_args = get_main_args_parser().parse_args()
    # 加载模型
    dfdetr = load_model('/data/lhm/DETR-like/exps/ecls/checkpoint0064.pth',main_args) # <--修改为自己加载模型的路径
  
    files = os.listdir("/data/datasets/Detection/coco/ood") # <--修改为待预测图片所在文件夹路径
 
    cn = 0
    waste=0
    for file in files:
        img_path = os.path.join("/data/datasets/Detection/coco/ood", file) # <--修改为待预测图片所在文件夹路径
        im = Image.open(img_path)
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        scores, boxes, waste_time, obj, ori_scores = detect(img, dfdetr, transform)
        plot_result(im, scores, boxes, obj, ori_scores , save_name=file, imshow=False, imwrite=True, save_txt=True)
 
        cn+=1 
        waste+=waste_time
        waste_avg = waste/cn
        print(waste_avg)