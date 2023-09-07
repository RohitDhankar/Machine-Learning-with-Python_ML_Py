
#

> CenterNet - point based Object detection 
> Doesnt mention - ANYWHERE - that you need not Label Images - before you implement a CenterNet based - Object detector ? OR DOES IT ? 

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

```
August 11th, 2020. MIT CSAIL

Abstract:
Objects are commonly thought of as axis-aligned boxes in an image. Even before deep learning, the best performing object detectors classified rectangular image regions. On one hand, this approach conveniently reduces detection to image classification. On the other hand, it has to deal with a nearly exhaustive list of image regions that do not contain any objects. In this talk, I'll present an alternative representation of objects: as points. I'll show how to build an object detector from a keypoint detector of object centers. The presented approach is both simpler and more efficient (faster and/or more accurate) than equivalent box-based detection systems. Our point-based detector easily extends to other tasks, such as object tracking, monocular or Lidar 3D detection, and pose estimation.

Most detectors, including ours, are usually trained on a single dataset and then evaluated in that same domain. However, it is unlikely that any user of an object detection system only cares about 80 COCO classes or 23 nuScenes vehicle categories in isolation. More likely than not, object classes needed in a down-stream system are either spread over different data-sources or not annotated at all. In the second part of this talk, I'll present a framework for learning object detectors on multiple different datasets simultaneously. We automatically learn the relationship between different objects annotations in different datasets and automatically merge them into common taxonomy. The resulting detector then reasons about the union of object classes from all datasets at once. This detector is also easily extended to unseen classes by fine-tuning it on a small dataset with novel annotations.

Bio:
Philipp is an Assistant Professor in the Department of Computer Science at the University of Texas at Austin. He received his Ph.D. in 2014 from the CS Department at Stanford University and then spent two wonderful years as a PostDoc at UC Berkeley. His research interests lie in Computer Vision, Machine learning, and Computer Graphics. He is particularly interested in deep learning, image understanding, and vision and action.

```
<br/>

#

> Objects as Points

[Objects_as_Points-XingyiZhou,DequanWang,PhilippKrähenbühl](https://arxiv.org/abs/1904.07850)

```
Computer Science > Computer Vision and Pattern Recognition
[Submitted on 16 Apr 2019 (v1), last revised 25 Apr 2019 (this version, v2)]
Objects as Points - Xingyi Zhou, Dequan Wang, Philipp Krähenbühl

    Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time. 

Comments: 	12 pages, 5 figures
Subjects: 	Computer Vision and Pattern Recognition (cs.CV)
Cite as: 	arXiv:1904.07850 [cs.CV]
  	(or arXiv:1904.07850v2 [cs.CV] for this version)
  	
https://doi.org/10.48550/arXiv.1904.07850
```

#

<br/>

#

