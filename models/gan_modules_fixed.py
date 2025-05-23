import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, img_size=128):
        super().__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Calculate number of upsampling blocks needed
        num_upsamples = int(torch.log2(torch.tensor(img_size)) - 2)
        
        # Initial block
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)  # Changed to inplace=True for better memory usage
        )
        
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        curr_channels = 512
        
        for i in range(num_upsamples - 2):
            block = nn.Sequential(
                nn.ConvTranspose2d(curr_channels, curr_channels // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(curr_channels // 2),
                nn.ReLU(inplace=True)  # Changed to inplace=True
            )
            self.upsample_blocks.append(block)
            curr_channels = curr_channels // 2
        
        # Final upsampling blocks
        self.penultimate = nn.Sequential(
            nn.ConvTranspose2d(curr_channels, curr_channels // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(curr_channels // 2),
            nn.ReLU(inplace=True)  # Changed to inplace=True
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(curr_channels // 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        # Initialize weights for better training
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        x = self.initial(z)
        for block in self.upsample_blocks:
            x = block(x)
        x = self.penultimate(x)
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=128):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Calculate number of downsampling blocks needed
        num_downsamples = int(torch.log2(torch.tensor(img_size)) - 2)
        
        # Initial block
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)  # Changed to inplace=True
        )
        
        # Downsampling blocks
        self.downsample_blocks = nn.ModuleList()
        curr_channels = 64
        
        for i in range(num_downsamples - 1):
            block = nn.Sequential(
                nn.Conv2d(curr_channels, curr_channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(curr_channels * 2),
                nn.LeakyReLU(0.2, inplace=True)  # Changed to inplace=True
            )
            self.downsample_blocks.append(block)
            curr_channels = curr_channels * 2
        
        # Final classification block
        self.final = nn.Sequential(
            nn.Conv2d(curr_channels, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize weights for better training
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        x = self.initial(x)
        for block in self.downsample_blocks:
            x = block(x)
        x = self.final(x)
        return x.view(-1, 1).squeeze(1)

# Improved gradient checkpointing implementation
def gradient_checkpointing_enable(self):
    """Enable gradient checkpointing to save VRAM at the cost of computation speed"""
    import torch.utils.checkpoint as checkpoint
    
    if isinstance(self, Generator):
        # Store original forward methods
        self._orig_initial_forward = self.initial.forward
        self._orig_penultimate_forward = self.penultimate.forward
        self._orig_final_forward = self.final.forward
        
        # Create non-lambda checkpoint wrappers
        def checkpoint_initial(x):
            return checkpoint.checkpoint(self._orig_initial_forward, x)
        
        def checkpoint_penultimate(x):
            return checkpoint.checkpoint(self._orig_penultimate_forward, x)
        
        def checkpoint_final(x):
            return checkpoint.checkpoint(self._orig_final_forward, x)
        
        self.initial.forward = checkpoint_initial
        self.penultimate.forward = checkpoint_penultimate
        self.final.forward = checkpoint_final
        
        # Store original forward methods for upsample blocks and replace
        for i, block in enumerate(self.upsample_blocks):
            block._orig_forward = block.forward
            # Create non-capturing function to avoid reference issues
            def make_checkpoint_fn(block):
                orig_fn = block.forward
                def checkpoint_fn(x):
                    return checkpoint.checkpoint(orig_fn, x)
                return checkpoint_fn
            
            self.upsample_blocks[i].forward = make_checkpoint_fn(block)
        
    elif isinstance(self, Discriminator):
        # Store original forward methods
        self._orig_initial_forward = self.initial.forward
        self._orig_final_forward = self.final.forward
        
        # Create non-lambda checkpoint wrappers
        def checkpoint_initial(x):
            return checkpoint.checkpoint(self._orig_initial_forward, x)
        
        def checkpoint_final(x):
            return checkpoint.checkpoint(self._orig_final_forward, x)
        
        self.initial.forward = checkpoint_initial
        self.final.forward = checkpoint_final
        
        # Store original forward methods for downsample blocks and replace
        for i, block in enumerate(self.downsample_blocks):
            block._orig_forward = block.forward
            # Create non-capturing function to avoid reference issues
            def make_checkpoint_fn(block):
                orig_fn = block.forward
                def checkpoint_fn(x):
                    return checkpoint.checkpoint(orig_fn, x)
                return checkpoint_fn
            
            self.downsample_blocks[i].forward = make_checkpoint_fn(block)
    
    print(f"[VRAM] Gradient checkpointing enabled for {self.__class__.__name__}")
    return True

def gradient_checkpointing_disable(self):
    """Disable gradient checkpointing, restoring normal operation"""
    if isinstance(self, Generator) and hasattr(self, '_orig_initial_forward'):
        # Restore original forward methods
        self.initial.forward = self._orig_initial_forward
        self.penultimate.forward = self._orig_penultimate_forward
        self.final.forward = self._orig_final_forward
        
        # Restore original forward methods for upsample blocks
        for block in self.upsample_blocks:
            if hasattr(block, '_orig_forward'):
                block.forward = block._orig_forward
                delattr(block, '_orig_forward')
        
        # Clean up stored methods
        delattr(self, '_orig_initial_forward')
        delattr(self, '_orig_penultimate_forward')
        delattr(self, '_orig_final_forward')
        
        print(f"[VRAM] Gradient checkpointing disabled for {self.__class__.__name__}")
        return True
    
    elif isinstance(self, Discriminator) and hasattr(self, '_orig_initial_forward'):
        # Restore original forward methods
        self.initial.forward = self._orig_initial_forward
        self.final.forward = self._orig_final_forward
        
        # Restore original forward methods for downsample blocks
        for block in self.downsample_blocks:
            if hasattr(block, '_orig_forward'):
                block.forward = block._orig_forward
                delattr(block, '_orig_forward')
        
        # Clean up stored methods
        delattr(self, '_orig_initial_forward')
        delattr(self, '_orig_final_forward')
        
        print(f"[VRAM] Gradient checkpointing disabled for {self.__class__.__name__}")
        return True
    
    return False

# Assign methods to classes
Generator.gradient_checkpointing_enable = gradient_checkpointing_enable
Generator.gradient_checkpointing_disable = gradient_checkpointing_disable
Discriminator.gradient_checkpointing_enable = gradient_checkpointing_enable
Discriminator.gradient_checkpointing_disable = gradient_checkpointing_disable
