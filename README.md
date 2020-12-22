# APTOS 2019 Blindness Detection
# DeepLearner 
This Rep. is "Selected Topics in Visual Recognition using Deep Learning" final project.

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
# getting start mmdectation
# https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md

# Pytorch 1.6.0  CUDA 10.2
pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install mmcv-full, we recommend you to install the pre-build package as below.
pip3 install mmcv-full==latest+torch1.7.0+cu110 -f https://download.openmmlab.com/mmcv/dist/index.html

# Clone the MMDetection repository.
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

# Install build requirements and then install MMDetection.
pip3 install -r requirements/build.txt
pip3 install -v -e .  # or "python setup.py develop"
```

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:

1.  [Dataset Preparation](#Dataset-Preparation)
2.  [Training](#Training)
3.  [Inference](#Inference)

## Dataset Preparation
### Prepare Images
After downloading images and spliting json file, the data directory is structured as:
```
 data/
  | +- train_images/
  | +- test_images/
  | -- pascal_train10.json
  | -- pascal_train90.json
  | -- test.json
```

#### Download Classes Image
Dataset: https://drive.google.com/drive/folders/1nglaZBJJ_Amonndw4nIVBh_UuCpp4gee?usp=sharing

Download and extract *train_images.zip* and *test_images.zip* to *data* directory.

#### Splited training and validation json used coco style
Use split8020.py to make train.txt .
```
$ python3 split8020.py
```

## Training
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


### Setting
You can setting detail Hyperparameters in [configs/mask_rcnn/mask_rcnn_r50_zino.py](https://github.com/linzino7/Fast_RCNN_mmdet/configs/mask_rcnn/mask_rcnn_r50_zino.py)

```
total_epochs = 100
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
work_dir = './work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco_posk_pretrain_200'
gpu_ids = range(1, 1)
```

### Train models
To train models, run following commands.
```
$ python3 mmdetection/tools/train.py configs/mask_rcnn/mask_rcnn_r50_zino.py
```
This project used Pre-train model. But according the mmdetection doc, it used pre-train backbone restnet-50 on ImageNet.

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time | mAP|
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | 
Fast_RCNN | 1x NVIDIA GTX 1080 | 1333, 800 | 100 | 6 hours | 0.


### Muti-GPU Training
I didn't test muti-GPU training.

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

