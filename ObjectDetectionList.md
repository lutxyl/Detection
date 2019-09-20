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

|   **Detector**   | **VOC07 (mAP@IoU=0.5)** | **VOC12 (mAP@IoU=0.5)** | **COCO (mAP@IoU=0.5:0.95)** | **Published In** |
|:------------:|:-------------------:|:-------------------:|:----------:|:------------:| 
|     R-CNN    |         58.5        |          -          |      -     |    CVPR'14   |
|    SPP-Net   |         59.2        |          -          |      -     |    ECCV'14   |
|  Fast R-CNN  |     70.0 (07+12)    |     68.4 (07++12)   |    19.7    |    ICCV'15   |
| Faster R-CNN |     73.2 (07+12)    |     70.4 (07++12)   |    21.9    |    NIPS'15   |
|    YOLO v1   |     66.4 (07+12)    |     57.9 (07++12)   |      -     |    CVPR'16   |
|   HyperNet   |     76.3 (07+12)    |    71.4 (07++12)    |      -     |    CVPR'16   |
|     OHEM     |     78.9 (07+12)    |    76.3 (07++12)    |    22.4    |    CVPR'16   |
|      SSD     |     76.8 (07+12)    |    74.9 (07++12)    |    31.2    |    ECCV'16   |
|     R-FCN    |     79.5 (07+12)    |    77.6 (07++12)    |    29.9    |    NIPS'16   |

##
## Network list (Updating)
The hyperlink directs to paper site, follows the official codes if the authors open sources. The OC means office code in Code* column, RC means recurrent code in Code* column. 

##
### 2014

|  **Model**   | **Paper Title**  |   **Keywords**   |   **Published**  |      **Code***     |   **Person liable***  |
|:-------------|:-----------------|:-----------------|:-----------------|:-------------------|:----------------------|
|  **R-CNN**   | Rich feature hierarchies for accurate object detection and semantic segmentation | ***CNNs, R-CNN*** | [CVPR' 14](https://arxiv.org/pdf/1311.2524.pdf) | [`[MATLAB OC]`](https://github.com/rbgirshick/rcnn) | Bin Wang |
| **OverFeat** | OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks | ***ConvNet, OverFeat*** | [ICLR' 14](https://arxiv.org/pdf/1312.6229.pdf) | [`[torch OC]`](https://github.com/sermanet/OverFeat) | Bin Wang |
| **MultiBox** | Scalable Object Detection using Deep Neural Networks | ***saliency-inspired neural network, class-agnostic bounding boxes*** |  [CVPR' 14](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf) | [`[caffe OC]`](https://github.com/google/multibox) | FangFang Cheng |
| **SPP-Net**  | Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition | ***Spatial Pyramid Pooling*** | [ECCV' 14](https://arxiv.org/pdf/1406.4729.pdf) | [`[caffe OC]`](https://github.com/ShaoqingRen/SPP_net) [`[tensorflow RC]`](https://github.com/peace195/sppnet) [`[keras RC]`](https://github.com/yhenon/keras-spp) | Bin Wang |

##
### 2015

|  **Model**   | **Paper Title**  |   **Keywords**   |   **Published**  |      **Code***     |   **Person liable***  |
|:-------------|:-----------------|:-----------------|:-----------------|:-------------------|:----------------------|
| **MR-CNN**   | Object detection via a multi-region & semantic segmentation-aware CNN model | ***multi-region, semantic segmentation-aware*** | [ICCV' 15](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Gidaris_Object_Detection_via_ICCV_2015_paper.pdf) | [`[caffe OC]`](https://github.com/gidariss/mrcnn-object-detection) | Bin Wang |
| **DeepBox**  | DeepBox: Learning Objectness with Convolutional Networks | ***DeepBox*** | [ICCV' 15](https://arxiv.org/pdf/1505.02146.pdf) | [`[caffe OC]`](https://github.com/weichengkuo/DeepBox) | Bin Wang |
| **AttentionNet** | AttentionNet: Aggregating Weak Directions for Accurate Object Detection | ***iterative classification*** | [ICCV' 15](https://arxiv.org/pdf/1506.07704.pdf) | - | Bin Wang |
| **Fast R-CNN** | Fast R-CNN | ***ROI Pooling, Feature Reuse*** | [ICCV' 15](https://arxiv.org/pdf/1504.08083.pdf) | [`[Caffe OC]`](https://github.com/rbgirshick/fast-rcnn) | Bin Wang |
| **DeepProposal** | DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers | ***DeepProposal, Cascade*** | [ICCV' 15](https://arxiv.org/pdf/1510.04445.pdf) | [`[matconvnet OC]`](https://github.com/aghodrati/deepproposal) | Bin Wang |
| **Faster R-CNN** | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | ***Real-time, RPN, Anchor*** | [NIPS' 15](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) | [`[caffe OC]`](https://github.com/rbgirshick/py-faster-rcnn) [`[tensorflow RC]`](https://github.com/endernewton/tf-faster-rcnn) [`[pytorch RC]`](https://github.com/jwyang/faster-rcnn.pytorch)  | Bin Wang |
##
### 2016

|  **Model**   | **Paper Title**  |   **Keywords**   |   **Published**  |      **Code***     |   **Person liable***  |
|:-------------|:-----------------|:-----------------|:-----------------|:-------------------|:----------------------|
| **YOLOV1** | You Only Look Once: Unified, Real-Time Object Detection  | ***1-stage, grid cell*** | [CVPR' 16](https://arxiv.org/pdf/1506.02640.pdf) | [`[C OC]`](https://pjreddie.com/darknet/yolo/) | Bin Wang |
| **HyperNet** | HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection | ***Hyper Feature*** | [CVPR' 16](https://arxiv.org/pdf/1604.00600.pdf) | - | Bin Wang |
| **OHEM** | Training Region-based Object Detectors with Online Hard Example Mining | ***online hard example mining*** | [CVPR' 16](https://arxiv.org/pdf/1604.03540.pdf) | [`[caffe OC]`](https://github.com/abhi2610/ohem) | Bin ang |
| **SSD** | SSD: Single Shot MultiBox Detector | ***1-stage, default-box, Hard negative mining, multi-feature maps*** | [ECCV' 16](https://arxiv.org/pdf/1512.02325.pdf) | [`[caffe OC]`](https://github.com/weiliu89/caffe/tree/ssd) [`[tensorflow RC]`](https://github.com/balancap/SSD-Tensorflow) [`[pytorch RC]`](https://github.com/amdegroot/ssd.pytorch)| Bin Wang |
| **R-FCN** | R-FCN: Object Detection via Region-based Fully Convolutional Networks  | ***PS ROI-Pooling, FCN*** | [NIPS](https://arxiv.org/pdf/1605.06409.pdf) | [`[caffe OC]`](https://github.com/daijifeng001/R-FCN) [`[caffe RC]`](https://github.com/YuwenXiong/py-R-FCN)| Bin Wang |


##
### 2017

|  **Model**   | **Paper Title**  |   **Keywords**   |   **Published**  |      **Code***     |   **Person liable***  |
|:-------------|:-----------------|:-----------------|:-----------------|:-------------------|:----------------------|
| **DSD** | DSSD : Deconvolutional Single Shot Detector | ***DDSD,object detection*** | [arXiv' 17](https://arxiv.org/pdf/1701.06659.pdf) | [`[caffe RC]`](https://github.com/chengyangfu/caffe/tree/dssd) | Fangfang Cheng |
| **YOLO v2** | YOLO9000: Better, Faster, Stronger | ***YOLO,real-time object detection*** |[CVPR'](https://arxiv.org/pdf/1612.08242.pdf) | [`[tensorflow OC]`](http://pjreddie.com/yolo9000/) | Fangfang Cheng |
| **RON** | RON: Reverse Connection with Objectness Prior Networks for Object Detection | ***RON,Prior networks,object detection*** | [CVPR' 17](https://arxiv.org/pdf/1707.01691.pdf) | [`[tensorflow OC]`](https://github.com/taokong/RON) | Fangfang Cheng |
| **DeNet** | DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling | ***Scalable Real-time Object Detection,Directed Sparse Sampling*** | [CVPR' 17](https://arxiv.org/pdf/1703.10295.pdf) | [`[Theano OC]`](https://github.com/lachlants/denet) | Fangfang Cheng |
| **CoupleNet** | CoupleNet: Coupling Global Structure with Local Parts for Object Detection | ***CoupleNet,object detection*** | [ICCV'17](https://arxiv.org/pdf/1708.02863.pdf) | [`[matlab OC]`](https://github.com/tshizys/CoupleNet) | Fangfang Cheng|

##
### 2018

|  **Model**   | **Paper Title**  |   **Keywords**   |   **Published**  |      **Code***     |   **Person liable***  |
|:-------------|:-----------------|:-----------------|:-----------------|:-------------------|:----------------------|
| **YOLO v3** | YOLOv3: An Incremental Improvement | ***YOLO*** | [arXiv' 18](https://pjreddie.com/media/files/papers/YOLOv3.pdf) | [`[tensorflow OC]`](https://pjreddie.com/darknet/yolo/) | Fangfang Cheng | 
| **SIN** | Structure Inference Net: Object Detection Using Scene-Level Context and Instance-Level Relationships | ***SIN,object detection,Scence-Level Context,Instance-Level Relationships*** | [CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Structure_Inference_Net_CVPR_2018_paper.pdf) | [`[tensorflow OC]`](https://github.com/choasup/SIN) | Fangfang Cheng |
| **RefineDet** | Sngle-Shot Refinement Neural Network for Object Detection | ***RefineDet,object detection*** | [CVPR' 18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf) | [`[caffe OC]`](https://github.com/sfzhang15/RefineDet) | Fangfang Cheng |
| **Cascade R-CNN** | Cascade R-CNN: Delving into High Quality Object Detection  | ***Cascade R-CNN,object detection*** | [CVPR' 18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf) | [`[caffe OC]`](https://github.com/zhaoweicai/cascade-rcnn) | Fangfang Cheng |
| **MLKP** | Multi-scale Location-aware Kernel Representation for Object Detection | ***MLKP,object detection*** | [CVPR' 18](https://arxiv.org/pdf/1804.00428.pdf ) |[`[matlab OC]`](https://github.com/Hwang64/MLKP) | Fangfang Cheng |

##
##
### 2019

|  **Model**   | **Paper Title**  |   **Keywords**   |   **Published**  |      **Code***     |   **Person liable***  |
|:-------------|:-----------------|:-----------------|:-----------------|:-------------------|:----------------------|
| **M2Det** | M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network | ***M2Det,single-shot object detector,multi-level feature pyramid network*** | [AAAI' 19](https://arxiv.org/pdf/1811.04533.pdf) | [`[pytarch OC]`](https://github.com/qijiezhao/M2Det) | Fangfang Cheng |
| **ExtremeNet** |  Bottom-up Object Detection by Grouping Extreme and Center Points | ***Bottom-up Object detection,a purely appearance-based keypoint estimation*** | [CVPR' 19](https://arxiv.org/pdf/1901.08043.pdf) | [`[pytorch OC]`](https://github.com/xingyizhou/ExtremeNet) | Fangfang Cheng | 
| **GFR** | Improving Object Detection from Scratch viaGated Feature Reuse | ***a parameter-efficient object detector,improve utilization of features on different scales*** | [BMVC' 19](https://arxiv.org/pdf/1712.00886v2.pdf) | [`[caffe OC]`](https://github.com/szq0214/GFR-DSOD) | Fangfang Cheng |
| **Libra R-CNN** | Libra R-CNN: Towards Balanced Learning for Object Detection | ***Libra R-CNN,IoU-banlacedd sampling,balanced feature pyramid,balaced L1 loss*** | [CVPR'19](https://arxiv.org/pdf/1904.02701v1.pdf) | [`[pytorch OC]`](https://github.com/OceanPang/Libra_R-CNN) | Fangfang Cheng |
| **HTC** | Hybrid Task Cascade for Instance Segmentation| ***HTC,Cascade R-CNN,Mask R-CNN*** | [CVPR'19](https://arxiv.org/pdf/1901.07518v2.pdf) | [`[pytorch OC]`](https://github.com/open-mmlab/mmdetectionf) | Fangfang Cheng |

##
## Link of datasets
*(please contact me if any of links offend you or any one disabled)*

Statistics of commonly used object detection datasets. The Table came from [this survey paper](https://arxiv.org/pdf/1809.02165v1.pdf).

<table>
<thead>
  <tr>
    <th rowspan=2>Challenge</th>
    <th rowspan=2 width=80>Object Classes</th>
    <th colspan=3>Number of Images</th>
    <th colspan=2>Number of Annotated Images</th>
  </tr>
  <tr>
    <th>Train</th>
    <th>Val</th>
    <th>Test</th>
    <th>Train</th>
    <th>Val</th>
  </tr>
</thead>
<tbody>

<!-- PASCAL VOC Object Detection Challenge -->
<tr><th colspan=7>PASCAL VOC Object Detection Challenge</th></tr>
<tr><td> VOC07 </td><td> 20 </td><td> 2,501 </td><td> 2,510 </td><td>  4,952 </td><td>   6,301 (7,844) </td><td>   6,307 (7,818) </td></tr>
<tr><td> VOC08 </td><td> 20 </td><td> 2,111 </td><td> 2,221 </td><td>  4,133 </td><td>   5,082 (6,337) </td><td>   5,281 (6,347) </td></tr>
<tr><td> VOC09 </td><td> 20 </td><td> 3,473 </td><td> 3,581 </td><td>  6,650 </td><td>   8,505 (9,760) </td><td>   8,713 (9,779) </td></tr>
<tr><td> VOC10 </td><td> 20 </td><td> 4,998 </td><td> 5,105 </td><td>  9,637 </td><td> 11,577 (13,339) </td><td> 11,797 (13,352) </td></tr>
<tr><td> VOC11 </td><td> 20 </td><td> 5,717 </td><td> 5,823 </td><td> 10,994 </td><td> 13,609 (15,774) </td><td> 13,841 (15,787) </td></tr>
<tr><td> VOC12 </td><td> 20 </td><td> 5,717 </td><td> 5,823 </td><td> 10,991 </td><td> 13,609 (15,774) </td><td> 13,841 (15,787) </td></tr>

<!-- ILSVRC Object Detection Challenge -->
<tr><th colspan=7>ILSVRC Object Detection Challenge</th></tr>
<tr><td> ILSVRC13 </td><td> 200 </td><td> 395,909 </td><td> 20,121 </td><td> 40,152 </td><td> 345,854 </td><td> 55,502 </td></tr>
<tr><td> ILSVRC14 </td><td> 200 </td><td> 456,567 </td><td> 20,121 </td><td> 40,152 </td><td> 478,807 </td><td> 55,502 </td></tr>
<tr><td> ILSVRC15 </td><td> 200 </td><td> 456,567 </td><td> 20,121 </td><td> 51,294 </td><td> 478,807 </td><td> 55,502 </td></tr>
<tr><td> ILSVRC16 </td><td> 200 </td><td> 456,567 </td><td> 20,121 </td><td> 60,000 </td><td> 478,807 </td><td> 55,502 </td></tr>
<tr><td> ILSVRC17 </td><td> 200 </td><td> 456,567 </td><td> 20,121 </td><td> 65,500 </td><td> 478,807 </td><td> 55,502 </td></tr>

<!-- MS COCO Object Detection Challenge -->
<tr><th colspan=7>MS COCO Object Detection Challenge</th></tr>
<tr><td> MS COCO15 </td><td> 80 </td><td>  82,783 </td><td> 40,504 </td><td> 81,434 </td><td> 604,907 </td><td> 291,875 </td></tr>
<tr><td> MS COCO16 </td><td> 80 </td><td>  82,783 </td><td> 40,504 </td><td> 81,434 </td><td> 604,907 </td><td> 291,875 </td></tr>
<tr><td> MS COCO17 </td><td> 80 </td><td> 118,287 </td><td>  5,000 </td><td> 40,670 </td><td> 860,001 </td><td>  36,781 </td></tr>
<tr><td> MS COCO18 </td><td> 80 </td><td> 118,287 </td><td>  5,000 </td><td> 40,670 </td><td> 860,001 </td><td>  36,781 </td></tr>

<!-- Open Images Object Detection Challenge -->
<tr><th colspan=7>Open Images Object Detection Challenge</th></tr>
<tr><td> OID18 </td><td> 500 </td><td> 1,743,042 </td><td> 41,620 </td><td> 125,436 </td><td> 12,195,144 </td><td> ― </td></tr>

  </tbody>
</table>

The papers related to datasets used mainly in Object Detection are as follows.

- **[PASCAL VOC]** The PASCAL Visual Object Classes (VOC) Challenge | **[IJCV' 10]** | [`[pdf]`](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)

- **[PASCAL VOC]** The PASCAL Visual Object Classes Challenge: A Retrospective | **[IJCV' 15]** | [`[pdf]`](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham15.pdf) | [`[link]`](http://host.robots.ox.ac.uk/pascal/VOC/)

- **[ImageNet]** ImageNet: A Large-Scale Hierarchical Image Database| **[CVPR' 09]** | [`[pdf]`](http://www.image-net.org/papers/imagenet_cvpr09.pdf)

- **[ImageNet]** ImageNet Large Scale Visual Recognition Challenge | **[IJCV' 15]** | [`[pdf]`](https://arxiv.org/pdf/1409.0575.pdf) | [`[link]`](http://www.image-net.org/challenges/LSVRC/)

- **[COCO]** Microsoft COCO: Common Objects in Context | **[ECCV' 14]** | [`[pdf]`](https://arxiv.org/pdf/1405.0312.pdf) | [`[link]`](http://cocodataset.org/)

- **[Open Images]** The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale | **[arXiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1811.00982v1.pdf) | [`[link]`](https://storage.googleapis.com/openimages/web/index.html)

- **[DOTA]** DOTA: A Large-scale Dataset for Object Detection in Aerial Images | **[CVPR' 18]** | [`[pdf]`](https://arxiv.org/pdf/1711.10398v3.pdf) | [`[link]`](https://captain-whu.github.io/DOTA/)

##
## Person liable
|   #   |                  **Name**                     |         **Mail***       |
|:------|:-------------------------------------------|:--------------------|
|   16  |[Xitong Chen](https://github.com/sleepercxt)|   375122362@qq.com  |
|   16  |[Jiaming Wang](-)|   -  |
|   18  |[Bin Wang](https://github.com/urbaneman)|   wangurbane@gmail.com  |
|   19  |[Fangfang Cheng](-)|   2475256748@qq.com  |

##