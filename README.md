# Project: Image super-resolution

## Overview
The target of this homework is  Image super-resolution.  
Super-Resolution convolutional neural networks have recently demonstrated high-quality restoration for single images. However, existing algorithms often require very deep architectures and long training times. Furthermore, current convolutional neural networks for super-resolution are unable to exploit features at multiple scales and weigh them equally, limiting their learning capability. In this exposition, I choose a compact and accurate super-resolution algorithm namely, Densely Residual Laplacian Network (DRLN) [1]. The proposed network employs cascading residual on the residual structure to allow the flow of low-frequency information to focus on learning high and mid-level features. In addition, deep supervision is achieved via the densely concatenated residual blocks settings, which also helps in learning from high-level complex features. Moreover, using Laplacian attention to model the crucial features to learn the inter and intra-level dependencies between the feature maps. Furthermore, comprehensive quantitative and qualitative evaluations on low-resolution, noisy low-resolution, and real historical image benchmark datasets illustrate that our DRLN algorithm performs favorably against the state-of-the-art methods visually and accurately.


## Hardware
The following specs were used to create the original solution.

* Ubuntu 18.04 LTS
* NVIDIA GeForce RTX 2080

## Download Official Image
You can download the training and testing dataset from Google Drive.  
https://drive.google.com/drive/u/0/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x

## Installation
* Linux or macOS with Python ≥ 3.6
* PyTorch 1.1.0
* CUDA9.0 
* cuDNN5.1
* OpenCV
* Numpy 1.19.2
* Tqdm 4.51.0
 
You need to create a dictory names 'checkpoints' to save chekpoint.  
And a dictory names 'output' to save images.
  
## Usage
Run train.py to start training.   
```
python Training.py
```

Test the model  
```
python Testing.py
```  

## Reference
https://github.com/saeed-anwar/DRLN  
https://ieeexplore.ieee.org/abstract/document/9185010  
https://arxiv.org/abs/1904.07523  

###### tags: `Image super-resolution` `Deep learning` `NCTU CS` `DRLN`
