- ViT -- 
> Input image ImageNet -- 224 X 224 
- split into 16 X 16 Blocks 
- get a FLAT Vector with -- 796 Dimensions 
- Linear Projection of Flattened Patches 
- 


#

<br/>

#

:star_struck:

> LSTM is dead. Long Live Transformers!
- YOUTUBE VIDEO -- https://www.youtube.com/watch?v=S27pHKBEp30&t=568s
- Leo Dirac (@leopd) talks about how LSTM models for Natural Language Processing (NLP) have been practically replaced by transformer-based models.  Basic background on NLP, and a brief history of supervised learning techniques on documents, from bag of words, through vanilla RNNs and LSTM.  Then there's a technical deep dive into how Transformers work with multi-headed self-attention, and positional encoding.  Includes sample code for applying these ideas to real-world projects.

- @8:50 -- LSTM - Transfer Learning not Ok 
- [@10:30](https://www.youtube.com/watch?v=S27pHKBEp30&t=630s)- Attention is all you need -- Multi Head Attention Mechanism -- 
- 

#

<br/>

#

Published as a conference paper at ICLR 2021

> AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE -- Alexey Dosovitskiy∗,†, Lucas Beyer∗, Alexander Kolesnikov∗, Dirk Weissenborn∗, Xiaohua Zhai∗, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby∗,† ∗equal technical contribution, †equal advising Google Research, Brain Team {adosovitskiy, neilhoulsby}@google.com

- ABSTRACT - While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited ...

- https://arxiv.org/pdf/2010.11929.pdf
- Short Name -- Vision_Transformers__AlexeyDosovitskiy_2010.11929.pdf

#

<br/>

#

> Transformers in Vision: A Survey
- https://arxiv.org/pdf/2101.01169.pdf


#

<br/>

#


> A Survey of Transformers - TIANYANG LIN, YUXIN WANG, XIANGYANG LIU, and XIPENG QIU∗, School of Computer
Science, Fudan University, China and Shanghai Key Laboratory of Intelligent Information Processing, Fudan
University, China

- ABSTRACT --    Transformers have achieved great success in many artificial intelligence fields, such as natural language
processing, computer vision, and audio processing. Therefore, it is natural to attract lots of interest from
academic and industry researchers. Up to the present, a great variety of Transformer variants (a.k.a. X-formers)
have been proposed, however, a systematic and comprehensive literature review on these Transformer variants
is still missing. In this survey, we provide a comprehensive review of various X-formers. We first briefly
introduce the vanilla Transformer and then propose a new taxonomy of X-formers. Next, we introduce the
various X-formers from three perspectives: architectural modification, pre-training, and applications. Finally,
we outline some potential directions for future research.

- https://arxiv.org/pdf/2106.04554.pdf

- Transformer Attention Modules --  Query-Key-Value

#

<br/>

#

#### Training Generative Adversarial Networks with Limited Data -- Tero Karras | NVIDIA
- https://arxiv.org/pdf/2006.06676.pdf

> AUTHORS -- Miika Aittala NVIDIA Janne Hellsten NVIDIA Samuli Laine NVIDIA Jaakko Lehtinen NVIDIA and Aalto University
Timo Aila NVIDIA

#### FOCUS ---> :star_struck:
> The key problem with small datasets is that the discriminator overfits to the training examples; its feedback to the generator becomes meaningless and training starts to diverge [2, 48 ]. In almost all areas of deep learning [40 ], dataset augmentation is the standard solution against overfitting. For

#### FOCUS ---> :star_struck:
With a 2k training set, the vast majority of the benefit came from pixel blitting and geometric transforms. Color transforms were modestly beneficial, while image-space filtering, noise, and cutout were not particularly useful. 

> Abstract -- Training generative adversarial networks (GAN) using too little data typically leads
to discriminator overfitting, causing training to diverge. We propose an adaptive
discriminator augmentation mechanism that significantly stabilizes training in
limited data regimes. The approach does not require changes to loss functions
or network architectures, and is applicable both when training from scratch and
when fine-tuning an existing GAN on another dataset. We demonstrate, on several
datasets, that good results are now possible using only a few thousand training
images, often matching StyleGAN2 results with an order of magnitude fewer
images. We expect this to open up new application domains for GANs. We also
find that the widely used CIFAR-10 is, in fact, a limited data benchmark, and
improve the record FID from 5.59 to 2.42.

> 1 Introduction
The increasingly impressive results of generative adversarial networks (GAN) [14 , 32 , 31 , 5 , 19 ,
20 , 21 ] are fueled by the seemingly unlimited supply of images available online. 

Still, it remains challenging to collect a large enough set of images for a specific application that places constraints
on subject type, image quality, geographical location, time period, privacy, copyright status, etc.

The difficulties are further exacerbated in applications that require the capture of a new, custom dataset: acquiring, processing, and distributing the ∼ 105 − 106 images required to train a modern high-quality, high-resolution GAN is a costly undertaking. This curbs the increasing use of generative models in fields such as medicine [ 47 ]. 

A significant reduction in the number of images required therefore has the potential to considerably help many applications.

The key problem with small datasets is that the discriminator overfits to the training examples; its feedback to the generator becomes meaningless and training starts to diverge [2, 48 ]. In almost all areas of deep learning [40 ], dataset augmentation is the standard solution against overfitting. For
example, training an image classifier under rotation, noise, etc., leads to increasing invariance to these
semantics-preserving distortions — a highly desirable quality in a classifier [ 17, 8, 9 ]. In contrast,
a GAN trained under similar dataset augmentations learns to generate the augmented distribution
[50 , 53 ]. In general, such “leaking” of augmentations to the generated samples is highly undesirable.
For example, a noise augmentation leads to noisy results, even if there is none in the dataset



#

<br/>

#

