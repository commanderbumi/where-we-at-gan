from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class JapanUKDataset(Dataset):
   def __init__(self, root_UK, root_japan, transform=None):
       self.root_UK = root_UK
       self.root_japan = root_japan
       self.transform = transform


       self.UK_images = os.listdir(root_UK)
       self.japan_images = os.listdir(root_japan)
       self.length_dataset = max(len(self.UK_images), len(self.japan_images)) # 1000, 1500
       self.UK_len = len(self.UK_images)
       self.japan_len = len(self.japan_images)


   def __len__(self):
       return self.length_dataset


   def __getitem__(self, index):
       UK_img = self.UK_images[index % self.UK_len]
       japan_img = self.japan_images[index % self.japan_len]


       UK_path = os.path.join(self.root_UK, UK_img)
       japan_path = os.path.join(self.root_japan, japan_img)


       UK_img = np.array(Image.open(UK_path).convert("RGB"))
       japan_img = np.array(Image.open(japan_path).convert("RGB"))


       if self.transform:
           augmentations = self.transform(image=UK_img, image0=japan_img)
           UK_img = augmentations["image"]
           japan_img = augmentations["image0"]


       return UK_img, japan_img

