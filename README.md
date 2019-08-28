
# RSCUP2019 object detection track 3th solution

## Introduction

2019遥感图像稀疏表征与智能分析竞赛初赛第三名方案

code_base: <https://github.com/open-mmlab/mmdetection>

### Major tricks

- [x] Mask RCNN  0.3541
- [x] Hybrid Task RCNN + deform conv   0.36633
- [ ] expand bbox 0.364
- [x] cascade score thresh adopt to 0.5 0.366
- [x] small number class augmentation 0.372
- [ ] cross_entropy weighted  0.369 
- [x] sync BN 0.376
- [x] IOU sampler 0.383
- [ ] pesudo label fine tune 0.362
- [ ] balanced sampler 0.369
- [ ] augmentation 0.40
- [x] 3 scale test 0.399
- [x] resnext101 0.41
- [x] scale2  finetune 0.43

## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.

### PrepareData

python ./tools/prepare_data.py OR Dataprepare.ipynb

data will generate in ./data/rscup/annotation/ and ./data/rscup/train

### Train

./tools/dist_train.sh  ./configs/rscup/htc_next_3s.py  <gpu_num>

### Test

./tools/dist_test.sh  ./configs/rscup/htc_next_3s.py  ./work_dirs/htc_next_3s/epoch*.pth  <gpu_num> \

--out  test.pkl

### Merge_result

python ./tools/merge_result.py



###  Convert2caffe

We achieved the converter from Hybrid Task cascade RCNN trained with mmdetection to Caffe. 

Please refer "  ".
