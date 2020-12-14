import os
from glob import glob

import torch
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

from models import get_se_resnet50_gem


MODEL_PATH = './model.bin'
TEST_IMAGE_PATH = 'test_images'
device = torch.device("cuda")

# Load model
model = get_se_resnet50_gem(pretrain=None)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Inference
test_images = glob(os.path.join(TEST_IMAGE_PATH, '*.png'))
predictions = []
for i, im_path in enumerate(test_images):
    print(i, '/', len(test_images))
    image = Image.open(im_path)
    image = image.resize((256, 256), resample=Image.BILINEAR)
    image = transforms.ToTensor()(image).to(device)
    output = model(image.unsqueeze(0))

    # Make submission
    predictions.append((os.path.splitext(im_path.split('/')[-1])[0], output.item()))

    if i == 10:
        break


submission = pd.DataFrame(predictions)
submission.columns = ['id_code','diagnosis']
submission.to_csv('submission_raw_value.csv', index=False)
submission.head()

submission.loc[submission.diagnosis < 0.7, 'diagnosis'] = 0
submission.loc[(0.7 <= submission.diagnosis) & (submission.diagnosis < 1.5), 'diagnosis'] = 1
submission.loc[(1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), 'diagnosis'] = 2
submission.loc[(2.5 <= submission.diagnosis) & (submission.diagnosis < 3.5), 'diagnosis'] = 3
submission.loc[3.5 <= submission.diagnosis, 'diagnosis'] = 4

submission['diagnosis'] = submission['diagnosis'].astype(int)
submission.to_csv('submission.csv', index=False)
submission.head()

