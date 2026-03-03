import os
from tqdm import tqdm
import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers import DDIMScheduler, DDIMPipeline

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    raise NotImplementedError('not recommended to run without GPU')

# Load the model and pipeline
model = UNet2DModel.from_pretrained('google/ddpm-celebahq-256').to(device)
scheduler = DDIMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(num_inference_steps=50)
pipeline = DDIMPipeline(unet=model, scheduler=scheduler)

# Directory to save images
save_dir = 'datasets/generated/ddpm-celebahq-256'
os.makedirs(save_dir, exist_ok=True)

# Config
num_images = 32000
batch_size = 64  # You can tune this depending on GPU memory

# Generate and save images
num_batches = (num_images + batch_size - 1) // batch_size

img_counter = 0
for _ in tqdm(range(num_batches), desc="Generating images"):
    current_batch_size = min(batch_size, num_images - img_counter)
    images = pipeline(batch_size=current_batch_size).images
    for img in images:
        img.save(os.path.join(save_dir, f'image_{img_counter:05d}.png'))
        img_counter += 1
