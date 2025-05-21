import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


TRAIN_DIR = "Regions/train"
VAL_DIR = "Regions/val"
BATCH_SIZE = 1
LEARNING_RATE = 2.5e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_J = "genJ.pth.tar"
CHECKPOINT_GEN_U = "genU.pth.tar"
CHECKPOINT_CRITIC_J = "criticJ.pth.tar"
CHECKPOINT_CRITIC_U = "criticU.pth.tar"


transforms = A.Compose(
    [
        A.Resize(width=384, height=156),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    is_check_shapes=False 
)

