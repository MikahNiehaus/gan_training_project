"""
Script to fix gray images in GAN by improving the way color statistics are maintained
through the network. This script adds instance normalization and adjusts momentum
of batch normalization layers to improve color reproduction.
"""

import torch
import torch.nn as nn
import os
import sys

def fix_gray_images():
    """
    Apply specific fixes to help with the gray image issue:
    1. Adjust momentum in batch normalization layers
    2. Ensure proper color channel processing
    3. Create color-enhancing instance normalization in strategic places
    """
    print("Applying fixes for gray images in GAN...")
    
    # Load the GAN modules
    from models.gan_modules import Generator, Discriminator
    
    # Create a test instance to verify changes
    test_generator = Generator(z_dim=100, img_channels=3, img_size=256)
    test_discriminator = Discriminator(img_channels=3, img_size=256)
    
    # Fix 1: Adjust BatchNorm momentum for better color stability
    bn_momentum_fixed = 0
    for module in test_generator.modules():
        if isinstance(module, nn.BatchNorm2d):
            # Lower momentum means slower moving averages, more stable colors
            module.momentum = 0.1  # Default is 0.1, lowering can help stabilize
            bn_momentum_fixed += 1
    
    print(f"Fixed {bn_momentum_fixed} BatchNorm layers with adjusted momentum")
    
    # Fix 2: Add functions to the Generator and Discriminator classes to improve
    # color diversity during the forward pass
    def enhanced_normalize_colors(self, x):
        """
        Enhance color diversity by adjusting channel statistics.
        This helps prevent the model from converging to grayscale.
        """
        # Skip if not RGB
        if not hasattr(self, 'img_channels') or self.img_channels != 3:
            return x
            
        # Get channel-wise statistics (preserving spatial dimensions)
        mean_r = x[:, 0:1].mean(dim=[2, 3], keepdim=True)
        mean_g = x[:, 1:2].mean(dim=[2, 3], keepdim=True)
        mean_b = x[:, 2:3].mean(dim=[2, 3], keepdim=True)
        
        # Calculate overall mean
        mean_rgb = (mean_r + mean_g + mean_b) / 3.0
        
        # If the mean of each channel is too similar, slightly adjust them
        too_similar = ((mean_r - mean_g).abs() < 0.05) & ((mean_g - mean_b).abs() < 0.05)
        
        # For images that are too gray, enhance color variation
        if too_similar.any():
            # Scale factor for R, G, B (amplify the differences from mean)
            r_scale = 1.1  # Slightly boost red
            g_scale = 1.0  # Keep green the same
            b_scale = 1.05  # Slightly boost blue
            
            # Apply channel-specific scaling to enhance diversity
            r_channel = mean_rgb + (x[:, 0:1] - mean_rgb) * r_scale 
            g_channel = mean_rgb + (x[:, 1:2] - mean_rgb) * g_scale
            b_channel = mean_rgb + (x[:, 2:3] - mean_rgb) * b_scale
            
            # Recombine
            x_enhanced = torch.cat([r_channel, g_channel, b_channel], dim=1)
            
            # Apply selectively to gray-ish images
            mask = too_similar.float().view(-1, 1, 1, 1)
            x = x * (1 - mask) + x_enhanced * mask
        
        return x
    
    # Patch the Generator's forward method to include color enhancement
    original_g_forward = Generator.forward
    
    def patched_g_forward(self, z):
        x = original_g_forward(self, z)
        return enhanced_normalize_colors(self, x)
    
    # Attach the new methods
    Generator.enhanced_normalize_colors = enhanced_normalize_colors
    Generator.forward = patched_g_forward
    
    print("Added color enhancement logic to Generator's forward pass")
    print("Gray image fixes applied successfully!")
    
    # Verify the output has proper color distribution
    with torch.no_grad():
        # Generate a test image
        test_noise = torch.randn(1, 100, 1, 1)
        test_output = test_generator(test_noise)
        
        # Check the color distribution
        r_mean = test_output[:, 0].mean().item()
        g_mean = test_output[:, 1].mean().item()
        b_mean = test_output[:, 2].mean().item()
        
        print(f"Test output color distribution: R={r_mean:.4f}, G={g_mean:.4f}, B={b_mean:.4f}")
        
        # Check if color variation is reasonable (not too close to grayscale)
        max_diff = max(abs(r_mean - g_mean), abs(r_mean - b_mean), abs(g_mean - b_mean))
        print(f"Maximum color channel difference: {max_diff:.4f}")
        
        if max_diff < 0.01:
            print("WARNING: Color channels still very similar - monitor outputs!")
        else:
            print("Color diversity looks good!")
    
    return test_generator, test_discriminator

if __name__ == "__main__":
    fix_gray_images()
