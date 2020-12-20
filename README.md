# DeepLearner

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
