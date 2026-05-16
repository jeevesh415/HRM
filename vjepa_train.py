import os
import argparse
import shutil
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
from models.muon_optimizer import Muon
from dataset.video_dataset import get_dataloader


def build_optimizer(model, config):
    """
    Build optimizer based on config.
    
    Supports:
      - 'adamw': Standard AdamW (default)
      - 'muon': Muon optimizer with Newton-Schulz orthogonalization
      - 'hybrid': Muon for weight matrices, AdamW for biases/norms
    """
    optimizer_type = config["training"].get("optimizer", "adamw")
    lr = float(config["training"]["lr"])
    
    if optimizer_type == "muon":
        # Muon for all parameters
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        optimizer = Muon(
            model.parameters(),
            lr=lr,
            momentum=config["training"].get("muon_momentum", 0.95),
            weight_decay=config["training"].get("weight_decay", 0.1),
            nesterov=True,
            ns_steps=config["training"].get("muon_ns_steps", 5),
            world_size=world_size,
            rank=rank,
        )
        print(f"Using Muon optimizer (lr={lr}, world_size={world_size})")
        
    elif optimizer_type == "hybrid":
        # Muon for 2D+ params (weight matrices), AdamW for 1D params (biases, norms)
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        muon_optimizer = Muon(
            muon_params,
            lr=lr,
            momentum=config["training"].get("muon_momentum", 0.95),
            weight_decay=config["training"].get("weight_decay", 0.1),
            nesterov=True,
            ns_steps=config["training"].get("muon_ns_steps", 5),
            world_size=world_size,
            rank=rank,
        )
        adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=lr,
            weight_decay=config["training"].get("weight_decay", 0.1),
        )
        
        optimizer = CombinedOptimizer([muon_optimizer, adamw_optimizer])
        print(f"Using hybrid Muon+AdamW optimizer (lr={lr}, muon_params={len(muon_params)}, adamw_params={len(adamw_params)})")
        
    else:
        # Standard AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config["training"].get("weight_decay", 0.01),
        )
        print(f"Using AdamW optimizer (lr={lr})")
    
    return optimizer


class CombinedOptimizer:
    """
    Wrapper to combine multiple optimizers into one.
    Used for the hybrid Muon+AdamW configuration.
    """
    def __init__(self, optimizers):
        self.optimizers = optimizers
    
    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)
    
    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups


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
        
    optimizer = build_optimizer(model, config)

    # 4. Data Manifold
    video_dir = "data"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        print(f"Created directory {video_dir}. Please add videos here.")
        import subprocess
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            print("ffmpeg not found; skipping synthetic video generation. Add videos manually to data/.")
        else:
            subprocess.run([
                ffmpeg_bin, '-f', 'lavfi', '-i', 'testsrc=duration=5:size=224x224:rate=15',
                os.path.join(video_dir, 'test_video.mp4'), '-y'
            ], capture_output=True, check=False)
        
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
    epochs = int(config.get("training", {}).get("epochs", 100))
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate dummy actions for testing physical interventions
            if "action" not in batch:
                batch["action"] = torch.randn(batch["video"].shape[0], action_dim, device=device)
            
            # Forward
            outputs = model(batch)
            
            # Loss Calculation (Advanced VICReg)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # EMA Update
            model.update_target_encoder()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
                # wandb.log({"loss": loss.item()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V-JEPA/HRM model")
    parser.add_argument(
        "--config",
        default="config/vjepa_micro.yaml",
        help="Path to YAML config file (e.g., config/vjepa_micro.yaml or config/vjepa_10b.yaml)",
    )
    args = parser.parse_args()
    train(args.config)
