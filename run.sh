

# Pre-train
# nohup python3 train_gem.py ~/Pretrained/train/  ~/Pretrained/trainLabels.csv cuda:1 --pretrain > pretrain.log &

# Fine-tune
nohup python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --model_path pretrain_weight/pretrain_model30.pth > finetune.log &
