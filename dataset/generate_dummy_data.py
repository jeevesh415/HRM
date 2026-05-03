import numpy as np
import cv2
import os

def generate_dummy_video(path, frames=32, res=(224, 224)):
    """Generates a simple video of a moving circle to test physical intuition."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10.0, res)
    
    x, y = 50, 50
    dx, dy = 5, 3
    
    for _ in range(frames):
        img = np.zeros((res[1], res[0], 3), dtype=np.uint8)
        cv2.circle(img, (x, y), 20, (0, 255, 0), -1)
        out.write(img)
        
        # Simple bounce physics
        x += dx
        y += dy
        if x <= 20 or x >= res[0]-20: dx *= -1
        if y <= 20 or y >= res[1]-20: dy *= -1
        
    out.release()
    print(f"Generated dummy video at {path}")

if __name__ == "__main__":
    generate_dummy_video("data/test_physics.mp4")
