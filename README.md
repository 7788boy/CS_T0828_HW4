# Project: Image super-resolution

## Overview
The target of this homework is Image super-resolution.  
Training on Tiny PASCAL VOC dataset with 20 common object classes and trying to detect each segmentation.

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
* OpenCV is optional and needed by demo and visualization
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

###### tags: `Image super-resolution` `Deep learning` `NCTU CS` `DRLN`
