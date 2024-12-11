# UN-DETR Promoting Objectness Learning via Joint Supervision for Unknown Object Detection

---

# Abstract 

Unknown Object Detection (UOD) aims to identify objects of unseen categories, differing from the traditional detection paradigm limited by the closed-world assumption. A key component of UOD is learning a generalized representation, i.e. objectness for both known and unknown categories to distinguish and localize objects from the background in a class-agnostic manner. However, previous methods obtain supervision signals for learning objectness in isolation from either localization or classification information, leading to poor performance for UOD. 
To address this issue, we propose a transformer-based UOD framework, UN-DETR. Upon this, we craft Instance Presence Score (IPS) to represent the probability of an object's presence. For the purpose of information complementarity, IPS employs a strategy of joint supervised learning, integrating attributes representing general objectness from the positional and the categorical latent space as supervision signals. To enhance IPS learning, we introduce a one-to-many assignment strategy to incorporate more supervision. Then, we propose Unbiased Query Selection to provide premium initial query vectors for the decoder. Additionally, we propose an IPS-based post-process strategy to filter redundant boxes and correct classification predictions for known and unknown objects. Finally, we pretrain the entire UN-DETR in an unsupervised manner, in order to obtain objectness prior. Our UN-DETR is comprehensively evaluated on multiple UOD and known detection benchmarks, demonstrating its effectiveness and achieving state-of-the-art performance

<img src=".\mdimg\main_structure.png" alt="main_structure" style="zoom:80%;" />

---

## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here]

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```


# Dataset Preparation

**PASCAL VOC**

Please put the corresponding json files in Google Cloud Disk into ./anntoations

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── JPEGImages
         ├── voc0712_train_all.json
         ├── voc0712_train_completely_annotation200.json
         └── val_coco_format.json

**COCO**

Please put the corresponding json files in Google Cloud Disk into ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
            ├── xxx (the original json files)
            ├── instances_val2017_coco_ood.json
            ├── instances_val2017_mixed_ID.json
            └── instances_val2017_mixed_OOD.json
         ├── train2017
         └── val2017

# Train

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
```

# Test

## Test_voc

```bash
sh eval_voc.sh
```
## Test_ood

```bash
sh eval_ood.sh
```
## Test_mix

```bash
sh eval_mix_id.sh
sh eval.mix_ood.sh
```

# Main Result

## On coco_ood

![image-20240820165900944](.\mdimg\image-20240820165900944.png)

## On coco-mixed

![image-20240820165920949](.\mdimg\image-20240820165920949.png)

## Visualization Result



<img src=".\mdimg\fig7nnn.png" alt="fig7nnn" style="zoom:80%;" />
