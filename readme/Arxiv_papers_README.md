
#

> CenterNet - point based Object detection - Anchorless 

#

> Doesnt mention - ANYWHERE - that you need not Label Images - before you implement a CenterNet based - Object detector ? OR DOES IT ? Is there a LABELED Training dataset required ? 

#

> The official Git Repo - presents some standard options to TRAIN with - COCO and KITTI for LIDAR generated point clouds.
So am presuming you will need LABELS to train against your own Custom Data ? 

- https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/readme/DATA.md

> Relevant extracts below, from the - readme/DATA.md , of the CenterNet 
You would not need the COCO Annotations if you did not need Class Labels or in this case Pre Labelled Images of Objects to be detected ...


```
    `-- |-- annotations
        |   |-- instances_train2017.json
        |   |-- instances_val2017.json
        |   |-- person_keypoints_train2017.json
        |   |-- person_keypoints_val2017.json
        |   |-- image_info_test-dev2017.json

```
> TRAIN CenterNet on your Custom Data as mentioned below in the --     CenterNet/readme/DEVELOP.md

- CenterNet/readme/DEVELOP.md -- https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/readme/DEVELOP.md?plain=1#L14


```
## New dataset
Basically there are three steps:

- Convert the dataset annotation to [COCO format](http://cocodataset.org/#format-data). Please refer to [src/tools/convert_kitti_to_coco.py](../src/tools/convert_kitti_to_coco.py) for an example to convert kitti format to coco format.
- Create a dataset intilization file in `src/lib/datasets/dataset`. In most cases you can just copy `src/lib/datasets/dataset/coco.py` to your dataset name and change the category information, and annotation path.
- Import your dataset at `src/lib/datasets/dataset_factory`.

## New task

You will need to add files to `src/lib/datasets/sample/`, `src/lib/datasets/trains/`, and `src/lib/datasets/detectors/`, which specify the data generation during training, the training targets, and the testing, respectively.

## New architecture
```
#

> How do we get the GROUND TRUTH data to benchmark against ? 

#

> below QUOTED as is from -- LEARN OPENCV .COM -- https://learnopencv.com/centernet-anchor-free-object-detection-explained/

- https://learnopencv.com/centernet-anchor-free-object-detection-explained/


```
Following are a few advantages of anchor free methods over anchor-based:

    Finding suitable anchor boxes (in shape and size) is crucial in training an excellent anchor-based object detection model. Finding suitable anchors is a complex problem and may need hyper-parameter tuning. 

    Using more anchors results in better accuracy in anchor-based object detection but using more anchors comes at a cost. The model needs more complex architecture, which leads to slower inference speed. 

    Anchor free object detection is more generalizable. It predicts objects as points that can easily be extended to key-points detection, 3D object detection, etc. However, the anchor-based object detection solution approach is limited to bounding box prediction.
```


- PR-241: Objects as Points--YouTube(https://www.youtube.com/watch?v=mDdpwe2xsT4)
- JiyanKang--YouTube(https://www.youtube.com/@JiyangKang)

- Philipp Krähenbühl - Point-based object detection--YouTube(https://www.youtube.com/watch?v=9vM6E6zoA84&t=2690s) 

#

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

> Variational Autoencoder

- [VAE_variational_autoencoder]
- [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)

```
Tutorial on Variational Autoencoders
CARL DOERSCH
Carnegie Mellon / UC Berkeley
August 16, 2016, with very minor revisions on January 3, 2021

Abstract
In just three years, Variational Autoencoders (VAEs) have emerged
as one of the most popular approaches to unsupervised learning of
complicated distributions. VAEs are appealing because they are built
on top of standard function approximators (neural networks), and
can be trained with stochastic gradient descent. VAEs have already
shown promise in generating many kinds of complicated data, in-
cluding handwritten digits [ 1, 2], faces [1, 3, 4], house numbers [5, 6],
CIFAR images [6 ], physical models of scenes [4], segmentation [ 7], and
predicting the future from static images [8]. This tutorial introduces the
intuitions behind VAEs, explains the mathematics behind them, and
describes some empirical behavior. No prior knowledge of variational
Bayesian methods is assumed.
Keywords: variational autoencoders, unsupervised learning, structured
prediction, neural networks
```

#

<br/>

#

- YOUTUBE 
- PROF . AHLAD KUMAR  PLAYLIST 
- https://www.youtube.com/watch?v=w8F7_rQZxXk&list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe


```
Quoted below as is from WIKI 

-- https://en.wikipedia.org/wiki/Variational_autoencoder

Overview of architecture and operation
A variational autoencoder is a generative model with a prior and noise distribution respectively. 
Usually such models are trained using the expectation-maximization meta-algorithm (e.g. probabilistic PCA, (spike & slab) sparse coding). 
Such a scheme optimizes a lower bound of the data likelihood, which is usually intractable, and in doing so requires the discovery of q-distributions, or variational posteriors. 
These q-distributions are normally parameterized for each individual data point in a separate optimization process. 

However, variational autoencoders use a neural network as an amortized approach to jointly optimize across data points. This neural network takes as input the data points themselves, and outputs parameters for the variational distribution. As it maps from a known input space to the low-dimensional latent space, it is called the encoder. 

```


