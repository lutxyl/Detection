# Object Detection
A collection of state-of-the-art detection architectures.

#### Update log
2019/9/16 update the structure of readme, reference from [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md). 

## Table of Contents
- [Performance table](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#performance-table)
- Papers
  - [2014](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#2014)
  - [2015](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#2015)
  - [2016](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#2016)
  - [2017](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#2017)
  - [2018](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#2018)
  - [2019](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#2019)
- [Dataset Papers](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#network-list-updating)
- [Link of Datesets](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#link-of-datasets)
- [Person liable](https://github.com/lutxyl/Detection/blob/master/ObjectDetectionList.md#person-liable)

##
## Performance table
FPS(Speed) index is related to the hardware spec(e.g. CPU, GPU, RAM, etc), so it is hard to make an equal comparison. The solution is to measure the performance of all models on hardware with equivalent specifications, but it is very difficult and time consuming. 

|   Detector   | VOC07 (mAP@IoU=0.5) | VOC12 (mAP@IoU=0.5) | COCO (mAP@IoU=0.5:0.95) | Published In |
|:------------:|:-------------------:|:-------------------:|:----------:|:------------:| 
|     R-CNN    |         58.5        |          -          |      -     |    CVPR'14   |

##
## Network list (Updating)
The hyperlink directs to paper site, follows the official codes if the authors open sources. The OC means office code in Code* column, RC means recurrent code in Code* column. 

##
### 2014

|  Model   | Paper Title  |   Keywords   |   Published  |      Code*     |   Person liable*  |
|:---------|:-------------|:-------------|:-------------|:---------------|:------------------|
|  R-CNN   | Rich feature hierarchies for accurate object detection and semantic segmentation | CNNs,R-CNN | [CVPR' 14](https://arxiv.org/pdf/1311.2524.pdf) | [MATLAB OC](https://github.com/rbgirshick/rcnn) | Bin Wang |
| OverFeat | OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks | ConvNet,OverFeat | [ICLR' 14] (https://arxiv.org/pdf/1312.6229.pdf) | [C++ OC](https://github.com/sermanet/OverFeat) | Bin Wang |
| MultiBox | Scalable Object Detection using Deep Neural Networks | saliency-inspired neural network, class-agnostic bounding boxes |  [CVPR' 14] (https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf) | [] (https://github.com/google/multibox) | FangFang Cheng |
|   |   |   |   |   |   |


##
### 2015

|  Model   | Paper Title  |   Keywords   |   Published  |      Code*     |   Person liable*  |
|:---------|:-------------|:-------------|:-------------|:---------------|:------------------|
|  Fast R-CNN   | Fast R-CNN | ROI Pooling, Feature Reuse | [ICCV' 15] (https://arxiv.org/pdf/1504.08083.pdf) | [Caffe OC] (https://github.com/rbgirshick/fast-rcnn) | Bin Wang |
|  Faster R-CNN   | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | Real-time, RPN, Anchor | [NIPS' 15] (https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) | [`caffe OC]`](https://github.com/rbgirshick/py-faster-rcnn) [`[tensorflow RC]`](https://github.com/endernewton/tf-faster-rcnn)  [`[pytorch RC]`](https://github.com/jwyang/faster-rcnn.pytorch)  | Bin Wang |


##
### 2016

|  Model   | Paper Title  |   Keywords   |   Published  |      Code*     |   Person liable*  |
|:---------|:-------------|:-------------|:-------------|:---------------|:------------------|
| |Part-Aligned Bilinear Representations for Person Re-Identification | |[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yumin_Suh_Part-Aligned_Bilinear_Representations_ECCV_2018_paper.pdf)|[PyTorch](https://github.com/yuminsuh/part_bilinear_reid)| Jiaming Wang |


##
### 2017

|  Model   | Paper Title  |   Keywords   |   Published  |      Code*     |   Person liable*  |
|:---------|:-------------|:-------------|:-------------|:---------------|:------------------|
| |Part-Aligned Bilinear Representations for Person Re-Identification | |[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yumin_Suh_Part-Aligned_Bilinear_Representations_ECCV_2018_paper.pdf)|[PyTorch](https://github.com/yuminsuh/part_bilinear_reid)| Jiaming Wang |

##
### 2018

|  Model   | Paper Title  |   Keywords   |   Published  |      Code*     |   Person liable*  |
|:---------|:-------------|:-------------|:-------------|:---------------|:------------------|
| |Part-Aligned Bilinear Representations for Person Re-Identification | |[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yumin_Suh_Part-Aligned_Bilinear_Representations_ECCV_2018_paper.pdf)|[PyTorch](https://github.com/yuminsuh/part_bilinear_reid)| Jiaming Wang |

##
### 2019

|  Model   | Paper Title  |   Keywords   |   Published  |      Code*     |   Person liable*  |
|:---------|:-------------|:-------------|:-------------|:---------------|:------------------|
| |Part-Aligned Bilinear Representations for Person Re-Identification | |[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yumin_Suh_Part-Aligned_Bilinear_Representations_ECCV_2018_paper.pdf)|[PyTorch](https://github.com/yuminsuh/part_bilinear_reid)| Jiaming Wang |

##
## Link of datasets
*(please contact me if any of links offend you or any one disabled)*

|     Name   |   Usage   |    #    |    Site    |    comments    |
|:-----------|:----------|:--------|:-----------|:---------------|
|Market1501|Train/Test|1501|[website](http://www.liangzheng.com.cn/Project/project_reid.html)|[Liang Zheng](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)|

##
## Person liable
|   #   |                  Name                      |         Mail*       |
|:------|:-------------------------------------------|:--------------------|
|   16  |[Xitong Chen](https://github.com/sleepercxt)|   375122362@qq.com  |

##
