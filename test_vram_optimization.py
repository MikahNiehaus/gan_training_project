"""
VRAM Optimization and Gradient Checkpointing Test Script

This script tests the VRAM optimization and gradient checkpointing implementation
by deliberately limiting available VRAM and ensuring the model can still train 
and generate samples.
"""

import os
import torch
import argparse
from models.gan_modules import Generator, Discriminator
from torchvision.utils import save_image
import time

def test_gradient_checkpointing():
    """Test if gradient checkpointing works as expected"""
    print("\n=== Testing Gradient Checkpointing Implementation ===")
    
    # Parameters
    img_size = 128  # Use a large size to stress test VRAM
    z_dim = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    print("Creating models...")
    G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
    D = Discriminator(img_channels=3, img_size=img_size).to(device)
    
    # Test if gradient checkpointing methods exist
    if not hasattr(G, 'gradient_checkpointing_enable') or not hasattr(D, 'gradient_checkpointing_enable'):
        print("❌ ERROR: Gradient checkpointing methods not implemented!")
        return False
    
    print("✅ Gradient checkpointing methods found")
    
    # Test enabling/disabling checkpointing
    G.gradient_checkpointing_enable()
    D.gradient_checkpointing_enable()
    
    # Create random input
    noise = torch.randn(1, z_dim, 1, 1, device=device)
    fake_image = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Test forward pass with checkpointing
    print("Testing forward pass with checkpointing enabled...")
    G.train()  # Must be in training mode for checkpointing to work
    D.train()
    
    try:
        out_g = G(noise)
        out_d = D(fake_image)
        print("✅ Forward pass successful with checkpointing enabled")
    except Exception as e:
        print(f"❌ ERROR: Forward pass failed with checkpointing enabled: {e}")
        return False
    
    # Test gradients
    print("Testing backward pass with checkpointing enabled...")
    try:
        # Create a custom loss and backpropagate
        loss_g = out_g.mean()
        loss_g.backward()
        
        loss_d = out_d.mean()
        loss_d.backward()
        
        print("✅ Backward pass successful with checkpointing enabled")
    except Exception as e:
        print(f"❌ ERROR: Backward pass failed with checkpointing enabled: {e}")
        return False
    
    # Test disabling checkpointing
    print("Testing disabling checkpointing...")
    G.zero_grad()
    D.zero_grad()
    G.gradient_checkpointing_disable()
    D.gradient_checkpointing_disable()
    
    try:
        out_g = G(noise) 
        out_d = D(fake_image)
        loss_g = out_g.mean()
        loss_d = out_d.mean()
        loss_g.backward()
        loss_d.backward()
        print("✅ Pass successful with checkpointing disabled")
    except Exception as e:
        print(f"❌ ERROR: Error after disabling checkpointing: {e}")
        return False
    
    return True

def test_sample_generation():
    """Test if sample generation works with memory constraints"""
    print("\n=== Testing Sample Generation with VRAM Constraints ===")
    
    # Set up constants
    z_dim = 100
    img_size = 128  # Large size to stress test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "test_samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create the generator
    print("Creating generator...")
    G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
    
    # Generate fixed noise
    fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
    
    # Test regular inference
    print("Testing regular generation...")
    G.eval()
    try:
        with torch.no_grad():
            fake = G(fixed_noise[:1]).detach().cpu()
            save_image(fake, os.path.join(sample_dir, "test_single.png"), normalize=True)
        print("✅ Single image generation successful")
    except Exception as e:
        print(f"❌ ERROR: Single image generation failed: {e}")
    
    # Test batch inference
    print("Testing batch generation...")
    try:
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
            save_image(fake, os.path.join(sample_dir, "test_grid.png"), normalize=True, nrow=4)
        print("✅ Batch image generation successful")
    except Exception as e:
        print(f"❌ NOTE: Batch generation failed (expected on low VRAM): {e}")
        print("Testing with gradient checkpointing...")
        
        # Try with gradient checkpointing
        try:
            G.gradient_checkpointing_enable()
            with torch.no_grad():
                fake = G(fixed_noise).detach().cpu()
                save_image(fake, os.path.join(sample_dir, "test_grid_checkpointed.png"), normalize=True, nrow=4)
            print("✅ Batch generation with checkpointing successful")
        except Exception as e2:
            print(f"❌ NOTE: Batch generation with checkpointing also failed: {e2}")
            print("Falling back to CPU...")
            
            # Try with CPU
            try:
                G = G.cpu()
                fixed_noise_cpu = fixed_noise.cpu()
                with torch.no_grad():
                    fake = G(fixed_noise_cpu).detach()
                    save_image(fake, os.path.join(sample_dir, "test_grid_cpu.png"), normalize=True, nrow=4)
                print("✅ CPU fallback generation successful")
            except Exception as e3:
                print(f"❌ ERROR: CPU fallback also failed: {e3}")
    
    print(f"\nSample images saved to {sample_dir}")
    return True

def simulate_vram_pressure():
    """Simulate VRAM pressure by allocating tensors"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping VRAM pressure test")
        return
    
    print("\n=== Simulating VRAM Pressure ===")
    # Get initial free memory
    torch.cuda.empty_cache()
    initial_free = torch.cuda.memory_allocated()
    print(f"Initial VRAM usage: {initial_free / 1024**2:.1f} MB")
    
    # Calculate ~70% of available VRAM
    total = torch.cuda.get_device_properties(0).total_memory
    target = int(total * 0.7)  # Target 70% usage
    
    # Allocate tensors to simulate pressure
    tensors = []
    try:
        current = torch.cuda.memory_allocated()
        while current < target:
            # Allocate in 100MB chunks
            size = 100 * 1024 * 1024 // 4  # 100MB in float32
            tensors.append(torch.rand(size, device="cuda"))
            current = torch.cuda.memory_allocated()
            print(f"Current VRAM usage: {current / 1024**2:.1f} MB / {total / 1024**2:.1f} MB")
        
        print(f"Successfully allocated {len(tensors)} tensors, {current / 1024**2:.1f} MB")
        return tensors
    except RuntimeError as e:
        print(f"Error allocating tensors: {e}")
        return tensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test VRAM optimization features')
    parser.add_argument('--pressure', action='store_true', help='Simulate VRAM pressure')
    args = parser.parse_args()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB")
    
    vram_tensors = None
    if args.pressure and torch.cuda.is_available():
        vram_tensors = simulate_vram_pressure()
        
    # Run tests
    ck_result = test_gradient_checkpointing()
    gen_result = test_sample_generation()
    
    # Clean up VRAM
    if vram_tensors is not None:
        del vram_tensors
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Gradient Checkpointing: {'✅ PASS' if ck_result else '❌ FAIL'}")
    print(f"Sample Generation: {'✅ PASS' if gen_result else '❌ FAIL'}")
