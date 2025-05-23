"""
Test script to check if GAN is producing colorful images instead of gray ones.
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from models.gan_modules import Generator

# Configuration
Z_DIM = 100
IMG_SIZE = 128  # Start with a lower resolution for testing
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join('data', 'gan_samples', 'test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_generator(num_samples=8):
    """
    Generate sample images and analyze their color variation to check if they are gray.
    """
    print(f"Testing generator at {IMG_SIZE}x{IMG_SIZE} resolution...")
    
    # Create a new generator
    G = Generator(z_dim=Z_DIM, img_channels=3, img_size=IMG_SIZE).to(DEVICE)
    
    # Try to load the latest checkpoint if available
    checkpoint_path = os.path.join('data', 'gan_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            if checkpoint.get('img_size', IMG_SIZE) == IMG_SIZE:
                G.load_state_dict(checkpoint['G'])
                print(f"Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"Resolution mismatch: checkpoint is for {checkpoint.get('img_size')}x{checkpoint.get('img_size')}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("No checkpoint found, using randomly initialized generator")
    
    # Set to evaluation mode
    G.eval()
    
    results = []
    
    # Generate multiple samples
    for i in range(num_samples):
        # Create random noise
        z = torch.randn(1, Z_DIM, 1, 1, device=DEVICE)
        
        # Generate image
        with torch.no_grad():
            fake = G(z).detach().cpu()
        
        # Save image
        output_path = os.path.join(OUTPUT_DIR, f'test_sample_{i}.png')
        save_image(fake, output_path, normalize=True)
        
        # Convert to numpy array and analyze color variation
        img_np = fake.squeeze(0).permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # Denormalize
        
        # Calculate color stats
        r_std = np.std(img_np[:, :, 0])
        g_std = np.std(img_np[:, :, 1])
        b_std = np.std(img_np[:, :, 2])
        
        # Calculate mean absolute difference between channels
        r_g_diff = np.mean(np.abs(img_np[:, :, 0] - img_np[:, :, 1]))
        r_b_diff = np.mean(np.abs(img_np[:, :, 0] - img_np[:, :, 2]))
        g_b_diff = np.mean(np.abs(img_np[:, :, 1] - img_np[:, :, 2]))
        
        # Calculate overall color diversity
        channel_diff = (r_g_diff + r_b_diff + g_b_diff) / 3
        std_avg = (r_std + g_std + b_std) / 3
        
        is_gray = channel_diff < 0.05  # Threshold for considering an image as gray
        
        results.append({
            'image': output_path,
            'r_std': r_std,
            'g_std': g_std,
            'b_std': b_std,
            'channel_diff': channel_diff,
            'is_gray': is_gray
        })
        
        print(f"Image {i}: {'GRAY' if is_gray else 'COLORFUL'}, "
              f"Channel diff: {channel_diff:.4f}, "
              f"Std: R={r_std:.4f}, G={g_std:.4f}, B={b_std:.4f}")
    
    # Print summary
    gray_count = sum(1 for r in results if r['is_gray'])
    print(f"\nSummary: {gray_count}/{num_samples} images are gray")
    print(f"Average channel difference: {sum(r['channel_diff'] for r in results) / num_samples:.4f}")
    print(f"Test images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    test_generator()
