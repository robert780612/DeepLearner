# APTOS 2019 Blindness Detection
> Team name : DeepLearner <br>
> Member : Robert, Lin, Zino

This Rep. is "Selected Topics in Visual Recognition using Deep Learning" final project in 2020.

## Hardware
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz
- 31 RAM
- NVIDIA GTX 1080 12G *1

## ENV
- CUDA 11.0
- efficientnet-pytorch    0.7.0
- torch                   1.7.0+cu110
- torchaudio              0.7.0
- torchvision             0.8.1+cu110


## Install env
```
# Pytorch 1.7.0  CUDA 11.2
pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

#  install requirements
pip3 install -r requirements.txt
```

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:

1.  [Dataset Preparation](#Dataset-Preparation)
2.  [Training](#Training)
3.  [Inference](#Inference)

## Dataset Preparation
### Download Kaggle Image
Data Link：
* [2015 Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
* [APTOS 2019 Blindness Detection images](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

Download image file to your disk.
Extract *aptos2019-blindness-detection.zip* and *diabetic-retinopathy-detection.zip* to *DeepLearner* and *Pretrained* directory.


#### Prepare Images
After downloading images and spliting json file, the data directory is structured as:

```
# 2015 Diabetic Retinopathy Detection images
 Pretrained/
  | +- sample/
  | +- train/
  | -- trainLabels.csv
  | -- sampleSubmission.csv
```

```
# APTOS 2019 Blindness Detection images
DeepLearner
 | +-train_images/
 | +-test_images/
 | --train.csv
 | --test.csv
 | --sample_submission.csv
```

## Training
### Train models
This project used Pre-train model. But according the mmdetection doc, it used pre-train backbone restnet-50 on ImageNet.

Train a model with image size 512, and train another model with image size 256.
Use the model 256 to make pseudo label on 2019 test dataset.
Finally, fine-tune model 512 on 2019 training + 2019 testing with pseudo label.

Pretrain on 2015 dataset
```
python3 train_gem.py ~/Pretrained/train/  ~/Pretrained/trainLabels.csv cuda:1 --img_size 512 --pretrain
```

Fine-tune on 2019 dataset
```
python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 512 --model_path pretrain_weight_512/pretrain_model30.pth
```

Third stage Fine-tune on 2019 train + test with pseudo label
```
python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 512 --model_path pretrain_weight_512/pretrain_model30.pth --additional_csv_path ./soft_pseudo_label.csv --additional_image_path ~/DeepLearner/test_images/
```

Make pseudo label
```
python3 train_gem.py ~/Pretrained/train/  ~/Pretrained/trainLabels.csv cuda:1 --img_size 256 --pretrain
python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 256 --model_path pretrain_weight_256/pretrain_model30.pth
python make_pseudo_label.py
```


The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time | 
------------ | ------------- | ------------- | ------------- | ------------- |
SE_restnet50 | 1x NVIDIA GTX 1080 | 256, 256 | 30 | 1 hours |



## Inference

### Inference single images
you need open mmtosummit.py to change model path and output name
```
$ python3 mmtosummit.py
```
## result
![](https://github.com/linzino7/Fast_RCNN_mmdet/blob/main/imgresult/in_2009_003123.jpg)
![](https://github.com/linzino7/Fast_RCNN_mmdet/blob/main/imgresult/in_2009_003938.jpg)

# mmdetection Modify

# Reference:

