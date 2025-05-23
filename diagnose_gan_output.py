import torch
from torchvision.utils import save_image
from models.gan_modules import Generator
import os

# Settings
Z_DIM = 100
IMG_SIZE = 64
SAMPLE_DIR = 'data/gan_samples/'
SAMPLE_PATH = os.path.join(SAMPLE_DIR, 'diagnostic_sample.png')

os.makedirs(SAMPLE_DIR, exist_ok=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. Generate with a freshly initialized model
    G = Generator(z_dim=Z_DIM, img_channels=3, img_size=IMG_SIZE).to(device)
    z = torch.randn(1, Z_DIM, 1, 1, device=device)
    with torch.no_grad():
        fake = G(z)
    save_image(fake, SAMPLE_PATH, normalize=True)
    print(f"[DIAG] Saved diagnostic sample from random Generator to {SAMPLE_PATH}")
    # 2. Print min/max/mean/std of output
    print(f"[DIAG] Output stats: min={fake.min().item():.3f}, max={fake.max().item():.3f}, mean={fake.mean().item():.3f}, std={fake.std().item():.3f}")
    # 3. Check for all-pink (R=1, G=B=0 or similar)
    r, g, b = fake[0,0], fake[0,1], fake[0,2]
    print(f"[DIAG] Channel means: R={r.mean().item():.3f}, G={g.mean().item():.3f}, B={b.mean().item():.3f}")
    if (r.mean() > 0.8 and g.mean() < 0.2 and b.mean() < 0.2):
        print("[DIAG] Output is strongly pink. This suggests a bug in initialization or normalization.")
    else:
        print("[DIAG] Output is not strongly pink. If you see pink in training, the bug is elsewhere.")

if __name__ == '__main__':
    main()
