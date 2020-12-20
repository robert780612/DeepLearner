

# Pre-train
#nohup python3 train_gem.py ~/Pretrained/train/  ~/Pretrained/trainLabels.csv cuda:1 --img_size 512 --pretrain > pretrain.log &

# Fine-tune
#nohup python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 512 --model_path pretrain_weight_512/pretrain_model30.pth > finetune.log &
# Soft label 
# nohup python3 train_gem.py ~/DeepLearner/train_images/  ./soft_pseudo_label.csv cuda:1 --img_size 256 --model_path pretrain_weight_256/pretrain_model30.pth > finetune.log &


# Pseudo label
#nohup python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 256 --model_path pretrain_weight_256/pretrain_model30.pth \
#	                   --additional_csv_path ./soft_pseudo_label.csv --additional_image_path ~/DeepLearner/test_images/ > stage3.log &

nohup python3 train_gem.py ~/DeepLearner/train_images/  ~/DeepLearner/train.csv cuda:1 --img_size 512 --model_path pretrain_weight_512/pretrain_model30.pth \
	                   --additional_csv_path ./soft_pseudo_label.csv --additional_image_path ~/DeepLearner/test_images/ > stage3_512.log &

# Train densenet
