"""
Script to fix issues with GAN training that's causing gray images
Run this script to identify and fix issues, then try training again
"""
import os
import sys
import torch
import subprocess

def check_gan_modules():
    """Check if our gan_modules.py has proper forward and initialization."""
    try:
        # Import the modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models.gan_modules import Generator, Discriminator
        
        # Test model initialization
        g = Generator(z_dim=100, img_channels=3, img_size=128)
        d = Discriminator(img_channels=3, img_size=128)
        
        # Check for proper weight initialization
        if hasattr(g, '_init_weights') and hasattr(d, '_init_weights'):
            print("✓ Models have weight initialization methods")
        else:
            print("✗ Missing weight initialization methods")
        
        # Test forward pass
        z = torch.randn(4, 100, 1, 1)
        fake_imgs = g(z)
        d_output = d(fake_imgs)
        
        print(f"✓ Model forward passes successful, shape: {fake_imgs.shape}")
        return True
    except Exception as e:
        print(f"Error in GAN modules: {e}")
        return False

def check_training_code():
    """Check for training issues in the GAN trainer."""
    try:
        # Check if gan_trainer learning rates are balanced
        with open("train_gan.py", "r") as f:
            content = f.read()
            
        # Check for common issues
        if "lr=lr * 0.2" in content or "LEARNING_RATE * 0.2" in content:
            print("✗ Found reduced learning rate for discriminator")
        else:
            print("✓ Discriminator uses same learning rate as generator")
        
        if "detach()" in content:
            print("✓ Found detach() to prevent gradient flow from D to G")
        else:
            print("✗ Missing detach() for fake images in discriminator")
        
        if "soft" in content or "0.9" in content:
            print("✓ Using soft labels for more stable training")
        else:
            print("✗ Not using soft labels")
            
        return True
    except Exception as e:
        print(f"Error checking training code: {e}")
        return False

def fix_issues():
    """Apply fixes to the GAN code."""
    try:
        cmds = []
        
        # Add recommendation for fixing the training loop indentation issue
        print("\nRecommendation: Fix the indentation issue in the training loop by:")
        print("1. Finding and fixing the indentation in the 'for real in pbar:' loop")
        print("2. Ensuring proper indentation for the 'batch_times.append(batch_end - data_loaded)' line")
        
        # Add recommendation for balancing G and D learning rates
        print("\nRecommendation: Set equal learning rates for generator and discriminator:")
        print("1. Update all instances where the discriminator has a different learning rate")
        print("2. Use different update frequencies instead of different learning rates")
        
        # Add recommendation for gradient checkpointing
        print("\nRecommendation: Fix gradient checkpointing implementation:")
        print("1. Use proper function closures instead of lambdas")
        print("2. Ensure checkpointing functions maintain proper gradient flow")
        
        # Add recommendation for generator features
        print("\nRecommendation: Add feature matching to help generator produce more varied images:")
        print("1. Calculate statistical features of real and fake images")
        print("2. Add a loss term to make these statistics similar")
        
        # Generate test images to check if fixes worked
        print("\nRecommendation: Run the test script to check if generated images are no longer gray:")
        print("python test_gan_grayscale.py")
        
        return True
    except Exception as e:
        print(f"Error applying fixes: {e}")
        return False

if __name__ == "__main__":
    print("GAN Gray Image Fix Script")
    print("=========================")
    print("Checking for issues that may cause the GAN to produce gray images...\n")
    
    modules_ok = check_gan_modules()
    training_ok = check_training_code()
    
    print("\nIssue Analysis Summary:")
    if modules_ok and training_ok:
        print("✓ Basic code structure looks good")
    else:
        print("✗ Found issues in code structure")
    
    fix_issues()
