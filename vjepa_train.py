import os
import yaml
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb

from models.vjepa.vjepa_model import VJEPA
from models.vjepa.losses import vicreg_loss
from dataset.video_dataset import get_dataloader

def train():
    # 1. Load 10B Config
    with open("config/vjepa_10b.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize Distributed Training (Mocked for single-process, ready for Multi-GPU)
    # dist.init_process_group("nccl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Model & Optimizer
    model = VJEPA(config["encoder"], config["predictor"], config["training"]["ema_momentum"]).to(device)
    # model = DDP(model) # FSDP is better for 10B, using DDP logic as placeholder
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])

    # 4. Data Manifold
    # Assuming a list of video paths is provided
    video_paths = ["path/to/video1.mp4", "path/to/video2.mp4"] # User will provide real paths
    dataloader = get_dataloader(video_paths, batch_size=config["training"]["batch_size"])

    # 5. Training Loop
    model.train()
    for epoch in range(100):
        for i, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward
            outputs = model(batch)
            
            # Loss Calculation (Advanced VICReg)
            # We apply VICReg to the predicted vs target latents
            loss = vicreg_loss(
                outputs["predicted"].reshape(-1, outputs["predicted"].shape[-1]), 
                outputs["target"].reshape(-1, outputs["target"].shape[-1]),
                sim_coeff=config["training"]["sim_coeff"],
                std_coeff=config["training"]["std_coeff"],
                cov_coeff=config["training"]["cov_coeff"]
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # EMA Update
            model.update_target_encoder()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
                # wandb.log({"loss": loss.item()})

if __name__ == "__main__":
    train()
