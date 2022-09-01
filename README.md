# Real_time_style_transfer
## Introduction
* This repository is implementation of Perceptual Losses for Real-Time Style Transfer and Super-Resolution
* You can do real time style transfer
* Difference from origin paper
  * for upsampling, use Conv2DTranspose with padding
  * for style, content loss, use squared sum instead norm
## Requirement
    python == 3.8
    tensorflow == 2.8.0
    opencv-python == 4.5.5
## Dataset
* use [COCO2014 Validation image dataset](https://cocodataset.org/#download)
* Dataset structure
    ```
  $root/
  ├── val2014
    ```
## Train
* Run
  ```
  python train.py --style_img --dataset_dir
  ```
* Args
  * style_img : target style image path
  * dataset_dir : dataset dir path
* Returns
  * ckpt : model checkpoint at every 100 iter. saved at "./ckpt" dir
  * model : final model. saved at "./models" dir
## Inference
* Run
  ```
  python infer.py --test_dir --model_path
  ```
* Arags
    * test_dir : test images dir path that you want style transfer
    * model_path : saved model path
* Returns
    * result : transfered images saved at "./reslut" dir
## Result
* Content image   
![Content](./asset/content.jpg)
* Transfer result   
![Result](./asset/result.png)
## Reference
[1] Perceptual Losses for Real-Time Style Transfer and Super-Resolution. Justin Johnson, Alexandre Alahi, Li Fei-Fei