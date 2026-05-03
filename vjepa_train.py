import os
import yaml
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from torch.utils.data import DataLoader
# import wandb

from models.vjepa.vjepa_model import VJEPA
from models.vjepa.losses import vicreg_loss
from dataset.video_dataset import get_dataloader

def train(config_path="config/vjepa_micro.yaml"):
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Starting training with config: {config_path}")
    
    # 2. Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Model & Optimizer
    action_dim = 128
    model = VJEPA(
        config["encoder"], 
        config["predictor"], 
        config["training"]["ema_momentum"],
        action_dim=action_dim
    ).to(device)
    
    # SOTA 10B Scaling: Fully Sharded Data Parallel (FSDP)
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = FSDP(
            model,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE
        )
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))

    # 4. Data Manifold
    video_dir = "data"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        print(f"Created directory {video_dir}. Please add videos here.")
        # Attempting to generate a dummy video if none exist
        import subprocess
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=224x224:rate=15', 
            os.path.join(video_dir, 'test_video.mp4'), '-y'
        ], capture_output=True)
        
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_paths:
        print("No videos found in data directory!")
        return

    print(f"Found {len(video_paths)} videos.")
    dataloader = get_dataloader(
        video_paths, 
        batch_size=config["training"]["batch_size"],
        resolution=(config["encoder"]["img_size"], config["encoder"]["img_size"]),
        patch_size=config["encoder"]["patch_size"],
        clip_len=config["encoder"]["max_t"] * config["encoder"]["patch_size"][0]
    )

    # 5. Training Loop
    model.train()
    for epoch in range(100):
        for i, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate dummy actions for testing physical interventions
            if "action" not in batch:
                batch["action"] = torch.randn(batch["video"].shape[0], action_dim, device=device)
            
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
