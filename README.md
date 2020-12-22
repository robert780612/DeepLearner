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

#### Pretrain on 2015 dataset
```
python3 train_gem.py ~/Pretrained/train/  ~/Pretrained/trainLabels.csv cuda:1 --img_size 512 --pretrain
```

#### Fine-tune on 2019 dataset
```
python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 512 --model_path pretrain_weight_512/pretrain_model30.pth
```

#### Third stage Fine-tune on 2019 train + test with pseudo label
```
python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 512 --model_path pretrain_weight_512/pretrain_model30.pth --additional_csv_path ./soft_pseudo_label.csv --additional_image_path ~/DeepLearner/test_images/
```

#### Make pseudo label
```
python3 train_gem.py ~/Pretrained/train/  ~/Pretrained/trainLabels.csv cuda:1 --img_size 256 --pretrain
python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 256 --model_path pretrain_weight_256/pretrain_model30.pth
python make_pseudo_label.py
```


The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time | 
------------ | ------------- | ------------- | ------------- | ------------- |
SE_restnet50 | 1x NVIDIA GTX 1080 | 256, 256 | 30 | 2 hours |



## Inference

### Inference and Make submission.csv
you need open mmtosummit.py to change model path and output name
```
$ python3 test.py
```
## result
### densenet121_bs64_90
![](https://github.com/robert780612/DeepLearner/blob/main/result/model_densenet121_bs64_90.PNG?raw=true)

### ensemble dense+seres+seres512 

![](https://github.com/robert780612/DeepLearner/blob/main/result/ensemble%20dense.seres.seres512_0.920265.png?raw=true)

### seresnet50 512  pseudo label  (BEST SCORE)
![](https://github.com/robert780612/DeepLearner/blob/main/result/pseudolabel.png?raw=true)

1. Train two SE-Resnet50 model which input size are 512 and 256 by 2015 and 2019 training data.
2. Used input size 256 of SE-Resnet50 to inference testing data and make pseudo-label.
3. Train input size 512 of SE-Resnet50 by 2019 training data and 2019 pseudo-label testing data.
3. Used input size 512 of SE-Resnet50 to make submission.csv.

# Reference:
- [Gold Medal Solutions](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108307) : Top 12 Ranking solutions. 
- [Rank 12 4uiiurz1](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection) 
- [pretrained-models](https://github.com/Cadene/pretrained-models.pytorch) : SE-Restnet 50  pretrained-models.
- [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252) : CVPR 2020 pseudo-label.
