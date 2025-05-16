import torch
from dataset import JapanUKDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_J, disc_U, gen_U, gen_J, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
 ):
    J_reals = 0
    J_fakes = 0
    loop = tqdm(loader, leave=True)
 
 
    for idx, (UK, japan) in enumerate(loop):
        UK = UK.to(config.DEVICE)
        japan = japan.to(config.DEVICE)
 
 
 	    # Train Discriminators H and Z
        with torch.amp.autocast(device_type=config.DEVICE.type if hasattr(config.DEVICE, "type") else "cpu", dtype=torch.float16):
            fake_japan = gen_J(UK)
            D_J_real = disc_J(japan)
 	 	    D_J_fake = disc_J(fake_japan.detach())
 	 	    J_reals += D_J_real.mean().item()
 	 	    J_fakes += D_J_fake.mean().item()
 	 	    D_J_real_loss = mse(D_J_real, torch.ones_like(D_J_real))
 	 	    D_J_fake_loss = mse(D_J_fake, torch.zeros_like(D_J_fake))
 	 	    D_J_loss = D_J_real_loss + D_J_fake_loss
 
 
 	 		fake_UK = gen_U(japan)
 	 	    D_U_real = disc_U(UK)
 	 	    D_U_fake = disc_U(fake_UK.detach())
 	 	    D_U_real_loss = mse(D_U_real, torch.ones_like(D_U_real))
 	 	    D_U_fake_loss = mse(D_U_fake, torch.zeros_like(D_U_fake))
 	 	    D_U_loss = D_U_real_loss + D_U_fake_loss
  
  
 	 	   # put it togethor
 	 		D_loss = (D_J_loss + D_U_loss) / 2
  
  
 	    opt_disc.zero_grad()
 	    d_scaler.scale(D_loss).backward()
 	    d_scaler.step(opt_disc)
 	    d_scaler.update()
  
  
 	    # Train Generators H and Z
 	    with torch.amp.autocast(device_type=config.DEVICE.type if hasattr(config.DEVICE, "type") else "cpu", dtype=torch.float16):
 	 	    # adversarial loss for both generators
 	 	    D_J_fake = disc_J(fake_japan)
 	 	    D_U_fake = disc_U(fake_UK)
 	 	    loss_G_J = mse(D_J_fake, torch.ones_like(D_J_fake))
 	 	    loss_G_U = mse(D_U_fake, torch.ones_like(D_U_fake))
   
   
 	 	    # cycle loss
 	 	    cycle_UK = gen_U(fake_japan)
 	 	    cycle_japan = gen_J(fake_UK)
 	 	    cycle_UK_loss = l1(UK, cycle_UK)
 	 		cycle_japan_loss = l1(japan, cycle_japan)
   
   
 	 	    # identity loss (remove these for efficiency if you set lambda_identity=0)
 	 	    identity_UK = gen_U(UK)
 	 	    identity_japan = gen_J(japan)
 	 	    identity_UK_loss = l1(UK, identity_UK)
 	 	    identity_japan_loss = l1(japan, identity_japan)
   
   
 	 	    # add all togethor
 	 	    G_loss = (
 	 	 	   loss_G_U
 	 	 	   + loss_G_J
 	 	 	   + cycle_UK_loss * config.LAMBDA_CYCLE
 	 	 	   + cycle_japan_loss * config.LAMBDA_CYCLE
 	 	 	   + identity_japan_loss * config.LAMBDA_IDENTITY
 	 	 	   + identity_UK_loss * config.LAMBDA_IDENTITY
 	 	    )
  
  
 	    opt_gen.zero_grad()
 	    g_scaler.scale(G_loss).backward()
 	    g_scaler.step(opt_gen)
 	    g_scaler.update()
  
  
 	    if idx % 200 == 0:
 	 		save_image(fake_japan * 0.5 + 0.5, f"saved_images/japan_{idx}.png")
 	 	    save_image(fake_UK * 0.5 + 0.5, f"saved_images/united_kingdom_{idx}.png")
 	 	
 	 		
  
 	    loop.set_postfix(J_real=J_reals / (idx + 1), J_fake=J_fakes / (idx + 1))




def main():
    disc_J = Discriminator(in_channels=3).to(config.DEVICE)
    disc_U = Discriminator(in_channels=3).to(config.DEVICE)
    gen_U = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_J = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
 		list(disc_J.parameters()) + list(disc_U.parameters()),
 	    lr=config.LEARNING_RATE,
 	    betas=(0.5, 0.999),
    )
 
 
    opt_gen = optim.Adam(
 		list(gen_U.parameters()) + list(gen_J.parameters()),
 	    lr=config.LEARNING_RATE,
 	    betas=(0.5, 0.999),
    )
 
 
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
 
 
    if config.LOAD_MODEL:
 	    load_checkpoint(
 	 		config.CHECKPOINT_GEN_J,
 	 	    gen_J,
 	 	    opt_gen,
 	 	    config.LEARNING_RATE,
 	    )
 	    load_checkpoint(
 	 	    config.CHECKPOINT_GEN_U,
 	 	    gen_U,
 	 	    opt_gen,
 	 	    config.LEARNING_RATE,
 	    )
 	    load_checkpoint(
 	 	    config.CHECKPOINT_CRITIC_J,
 	 	    disc_J,
 	 	    opt_disc,
 	 	    config.LEARNING_RATE,
 	    )
 	    load_checkpoint(
 	 	    config.CHECKPOINT_CRITIC_U,
 	 	    disc_U,
 	 	    opt_disc,
 	 	    config.LEARNING_RATE,
 	    )
 
 
    dataset = JapanUKDataset(
 	    root_japan=config.TRAIN_DIR + "/Japan",
 	    root_UK=config.TRAIN_DIR + "/United Kingdom",
 	    transform=config.transforms,
    )
    val_dataset = JapanUKDataset(
 	    root_japan=config.VAL_DIR + "/Japan",
 	    root_UK=config.VAL_DIR + "/United Kingdom",
 	    transform=config.transforms,
    )
    val_loader = DataLoader(
 	    val_dataset,
 	    batcJ_size=1,
 	    shuffle=False,
 	    pin_memory=True,
    )
    loader = DataLoader(
 		dataset,
 		batcJ_size=config.BATCJ_SIZE,
 		shuffle=True,
 		num_workers=config.NUM_WORKERS,
 		pin_memory=True,
    )
    if hasattr(config.DEVICE, "type") and config.DEVICE.type == "cuda":
 		g_scaler = torch.cuda.amp.GradScaler()
 		d_scaler = torch.cuda.amp.GradScaler()
    else:
 	    # On MPS or CPU, we'll use a dummy scaler that just does the operations without scaling
 	    class DummyScaler:
 	 	    def scale(self, loss):
 	 	 		return loss
 	 	    def step(self, optimizer):
 	 	 		optimizer.step()
 	 	    def update(self):
 	 	 		pass
 	 	    def unscale_(self, optimizer):
 	 	 		pass
 	   
 	    g_scaler = DummyScaler()
 	    d_scaler = DummyScaler()
 
 
    for epoch in range(config.NUM_EPOCHS):
 	    train_fn(
 	 	   disc_J,
 	 	   disc_U,
 	 	   gen_U,
 	 	   gen_J,
 	 	   loader,
 	 	   opt_disc,
 	 	   opt_gen,
 	 	   L1,
 	 	   mse,
 	 	   d_scaler,
 	 	   g_scaler,
 	    )
  
  
 	    if config.SAVE_MODEL:
 	 	   save_checkpoint(gen_J, opt_gen, filename=config.CHECKPOINT_GEN_J)
 	 	   save_checkpoint(gen_U, opt_gen, filename=config.CHECKPOINT_GEN_U)
 	 	   save_checkpoint(disc_J, opt_disc, filename=config.CHECKPOINT_CRITIC_J)
 	 	   save_checkpoint(disc_U, opt_disc, filename=config.CHECKPOINT_CRITIC_U)


if __name__ == "__main__":
   main()

