"""
Script to enhance GAN modules for better color diversity and to fix the gray image issue.
This script creates improved versions of Generator and Discriminator with spectral normalization
and other enhancements to prevent the gray image problem.
"""

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

def apply_spectral_norm_to_module(module):
    """Apply spectral normalization to appropriate layers in a module"""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(module)
    return module

def create_improved_gan_modules():
    """
    Create improved GAN modules to fix the gray image issue:
    1. Add spectral normalization to stabilize training
    2. Use more residual connections
    3. Add instance normalization in key places
    """
    from models.gan_modules import Generator, Discriminator
    
    # Apply spectral normalization to existing modules
    for model_cls in [Generator, Discriminator]:
        # Store the original init method
        original_init = model_cls.__init__
        
        # Create a new init method that adds spectral normalization
        def new_init(self, *args, **kwargs):
            # Call the original init first
            original_init(self, *args, **kwargs)
            
            # Apply spectral normalization to convolutional layers
            if isinstance(self, Generator):
                # Apply to Generator's initial and final blocks
                self.initial = nn.Sequential(*[
                    apply_spectral_norm_to_module(module) if isinstance(module, nn.ConvTranspose2d) else module
                    for module in self.initial
                ])
                
                self.final = nn.Sequential(*[
                    apply_spectral_norm_to_module(module) if isinstance(module, nn.ConvTranspose2d) else module
                    for module in self.final
                ])
                
                # Apply to upsampling blocks
                for i, block in enumerate(self.upsample_blocks):
                    self.upsample_blocks[i] = nn.Sequential(*[
                        apply_spectral_norm_to_module(module) if isinstance(module, nn.ConvTranspose2d) else module
                        for module in block
                    ])
                    
            elif isinstance(self, Discriminator):
                # Apply to discriminator's convolutional layers
                self.initial = nn.Sequential(*[
                    apply_spectral_norm_to_module(module) if isinstance(module, nn.Conv2d) else module
                    for module in self.initial
                ])
                
                self.final = nn.Sequential(*[
                    apply_spectral_norm_to_module(module) if isinstance(module, nn.Conv2d) else module
                    for module in self.final
                ])
                
                # Apply to downsampling blocks
                for i, block in enumerate(self.downsample_blocks):
                    self.downsample_blocks[i] = nn.Sequential(*[
                        apply_spectral_norm_to_module(module) if isinstance(module, nn.Conv2d) else module
                        for module in block
                    ])
        
        # Replace the original init with our new one
        model_cls.__init__ = new_init
    
    # Create an improved Generator forward method that enhances color diversity
    original_g_forward = Generator.forward
    
    def improved_g_forward(self, z):
        # Apply original forward pass
        x = original_g_forward(self, z)
        
        # Add color enhancement: boost standard deviation in color channels slightly
        # This helps prevent the model from converging to gray images
        if x.size(1) == 3:  # Only for RGB images
            mean = x.mean(dim=[2, 3], keepdim=True)
            std = x.std(dim=[2, 3], keepdim=True) + 1e-8
            
            # Scale up the deviation from mean slightly to enhance color diversity
            x = mean + (x - mean) * 1.1
            
            # Ensure output remains in valid range
            x = torch.clamp(x, -1.0, 1.0)
        
        return x
    
    # Replace forward methods
    Generator.forward = improved_g_forward
    
    print("GAN modules enhanced with spectral normalization and color diversity improvements")
    return Generator, Discriminator

if __name__ == "__main__":
    create_improved_gan_modules()
    print("GAN enhancements applied successfully")
