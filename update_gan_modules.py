"""
Update script to switch GANTrainer to use improved GAN modules
"""
import os
import argparse

def update_imports():
    """Update imports in train_gan.py to use improved modules"""
    with open('train_gan.py', 'r') as f:
        content = f.read()
    
    # Replace the import statement
    updated_content = content.replace(
        "from models.gan_modules import Generator, Discriminator",
        "from models.gan_modules_improved import Generator, Discriminator"
    )
    
    with open('train_gan.py', 'w') as f:
        f.write(updated_content)
    
    print("✓ Updated imports in train_gan.py to use improved GAN modules")

def update_gan_trainer_init():
    """Update GANTrainer.__init__ to use spectral normalization and better parameters"""
    with open('train_gan.py', 'r') as f:
        content = f.read()
    
    # Find the GANTrainer.__init__ method
    init_start = content.find("def __init__(self, img_size=128, z_dim=100, lr=2e-4, batch_size=32, device='cpu', use_neighbor_penalty=True, use_distortion=True, reference_image_path=None, fixed_seed=1234):")
    if init_start == -1:
        print("✗ Could not find GANTrainer.__init__ method")
        return
    
    # Find the lines creating G and D models
    g_model_line = content.find("self.G = Generator", init_start)
    d_model_line = content.find("self.D = Discriminator", init_start)
    
    if g_model_line == -1 or d_model_line == -1:
        print("✗ Could not find Generator and Discriminator instantiation")
        return
    
    # Update the model instantiation to include spectral normalization
    updated_content = content[:g_model_line] + \
        "self.G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size, use_spectral_norm=True).to(device)\n" + \
        "        self.D = Discriminator(img_channels=3, img_size=img_size, use_spectral_norm=True).to(device)" + \
        content[content.find("\n", d_model_line):]
    
    with open('train_gan.py', 'w') as f:
        f.write(updated_content)
    
    print("✓ Updated GANTrainer to use spectral normalization")

def update_checkpoint_methods():
    """Update checkpoint methods to use the new enable_checkpointing methods"""
    with open('train_gan.py', 'r') as f:
        content = f.read()
    
    # Replace the old method calls with the new ones
    updated_content = content.replace(
        "self.G.gradient_checkpointing_enable()",
        "self.G.enable_checkpointing()"
    ).replace(
        "self.D.gradient_checkpointing_enable()",
        "self.D.enable_checkpointing()"
    )
    
    with open('train_gan.py', 'w') as f:
        f.write(updated_content)
    
    print("✓ Updated checkpoint methods to use new API")

def fix_all():
    """Apply all fixes"""
    update_imports()
    update_gan_trainer_init()
    update_checkpoint_methods()
    print("\nAll updates applied. The GAN should now produce more colorful images.")
    print("To test, run 'python test_gan_grayscale.py' and then train with 'python train_gan.py'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update GAN modules to prevent gray image generation")
    parser.add_argument("--skip-confirm", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()
    
    if args.skip_confirm or input("Apply all updates to fix gray image generation? (y/n): ").lower() == 'y':
        fix_all()
    else:
        print("Updates cancelled.")
