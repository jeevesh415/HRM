import torch
import subprocess
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import random
from typing import Tuple, List, Optional

class AdvancedVideoDataset(IterableDataset):
    """
    Advanced Spatio-Temporal Video Data Manifold.
    Implements 3D Block Masking and Continuous Time Deltas for Neural ODEs.
    """
    def __init__(self, 
                 video_paths: List[str], 
                 resolution: Tuple[int, int] = (224, 224),
                 clip_len: int = 16,
                 frame_rate: int = 15,
                 patch_size: Tuple[int, int, int] = (2, 16, 16),
                 mask_ratio: float = 0.6):
        super().__init__()
        self.video_paths = video_paths
        self.resolution = resolution
        self.clip_len = clip_len
        self.frame_rate = frame_rate
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Grid dimensions for patches
        self.t_p = clip_len // patch_size[0]
        self.h_p = resolution[0] // patch_size[1]
        self.w_p = resolution[1] // patch_size[2]

    def _get_video_stream(self, path):
        # ffmpeg command for raw video streaming
        cmd = [
            'ffmpeg', '-i', path,
            '-vf', f'fps={self.frame_rate},scale={self.resolution[0]}:{self.resolution[1]}',
            '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _generate_3d_block_mask(self):
        """
        Generates a 3D block mask for V-JEPA training.
        """
        mask = torch.zeros(self.t_p, self.h_p, self.w_p, dtype=torch.bool)
        total_patches = self.t_p * self.h_p * self.w_p
        num_masked = int(total_patches * self.mask_ratio)
        
        # Multi-block masking strategy
        while mask.sum() < num_masked:
            # Random block dimensions
            bt = random.randint(1, self.t_p // 2)
            bh = random.randint(self.h_p // 4, self.h_p // 2)
            bw = random.randint(self.w_p // 4, self.w_p // 2)
            
            # Random block origin
            ot = random.randint(0, self.t_p - bt)
            oh = random.randint(0, self.h_p - bh)
            ow = random.randint(0, self.w_p - bw)
            
            mask[ot:ot+bt, oh:oh+bh, ow:ow+bw] = True
            
        return mask.view(-1)

    def __iter__(self):
        random.shuffle(self.video_paths)
        for path in self.video_paths:
            process = self._get_video_stream(path)
            frame_size = self.resolution[0] * self.resolution[1] * 3
            
            frames = []
            while True:
                raw_frame = process.stdout.read(frame_size)
                if not raw_frame: break
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(self.resolution[1], self.resolution[0], 3)
                frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
                
                if len(frames) == self.clip_len:
                    video_clip = torch.stack(frames) # (T, C, H, W)
                    
                    # Generate 3D Mask
                    mask = self._generate_3d_block_mask()
                    
                    # Continuous Time Deltas (1/fps)
                    # For Neural ODE, we provide the timestamp for each frame group
                    delta_t = torch.tensor([1.0 / self.frame_rate * self.patch_size[0]], dtype=torch.float32)
                    
                    # Multimodal Grounding Hook (Placeholder for Audio/Tactile)
                    multimodal_context = {
                        "audio": torch.zeros(1), # Future integration
                        "tactile": torch.zeros(1)
                    }
                    
                    yield {
                        "video": video_clip,
                        "mask": mask,
                        "delta_t": delta_t,
                        "multimodal": multimodal_context
                    }
                    frames = [] # Reset for next clip
                    
            process.terminate()

def get_dataloader(video_paths, batch_size=1, **kwargs):
    dataset = AdvancedVideoDataset(video_paths, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0) # FFMPEG pipe doesn't like multi-worker easily
