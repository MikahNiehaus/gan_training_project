"""
Enhanced versions of GAN modules with improvements to prevent gray image issues
Key improvements:
1. Better gradient flow
2. Enhanced generator architecture with spectral normalization
3. Improved weight initialization
4. Support for advanced checkpointing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNorm:
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        height = weight.size(0)
        
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.matmul(u, weight.view(height, -1)), dim=0, eps=self.eps)
            u = F.normalize(torch.matmul(weight.view(height, -1).t(), v), dim=0, eps=self.eps)
        
        sigma = torch.dot(u, torch.matmul(weight.view(height, -1), v))
        weight = weight / sigma
        return weight

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = getattr(module, name)
        height = weight.size(0)
        
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        
        delattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_buffer(name + '_u', u)
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, img_size=128, features_g=64, use_spectral_norm=True):
        super().__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.features_g = features_g
        self.use_spectral_norm = use_spectral_norm
        
        # Calculate number of upsampling blocks needed
        num_upsamples = int(torch.log2(torch.tensor(img_size)) - 2)
        
        # Initial block
        if use_spectral_norm:
            self.initial = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(z_dim, features_g * 8, 4, 1, 0, bias=False)),
                nn.BatchNorm2d(features_g * 8),
                nn.ReLU(inplace=True)
            )
        else:
            self.initial = nn.Sequential(
                nn.ConvTranspose2d(z_dim, features_g * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features_g * 8),
                nn.ReLU(inplace=True)
            )
        
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        curr_features = features_g * 8
        
        for i in range(num_upsamples - 2):
            if use_spectral_norm:
                block = nn.Sequential(
                    spectral_norm(nn.ConvTranspose2d(curr_features, curr_features // 2, 4, 2, 1, bias=False)),
                    nn.BatchNorm2d(curr_features // 2),
                    nn.ReLU(inplace=True)
                )
            else:
                block = nn.Sequential(
                    nn.ConvTranspose2d(curr_features, curr_features // 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(curr_features // 2),
                    nn.ReLU(inplace=True)
                )
            self.upsample_blocks.append(block)
            curr_features = curr_features // 2
        
        # Final upsampling blocks
        if use_spectral_norm:
            self.penultimate = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(curr_features, curr_features // 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(curr_features // 2),
                nn.ReLU(inplace=True)
            )
            
            self.final = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(curr_features // 2, img_channels, 4, 2, 1, bias=False)),
                nn.Tanh()
            )
        else:
            self.penultimate = nn.Sequential(
                nn.ConvTranspose2d(curr_features, curr_features // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(curr_features // 2),
                nn.ReLU(inplace=True)
            )
            
            self.final = nn.Sequential(
                nn.ConvTranspose2d(curr_features // 2, img_channels, 4, 2, 1, bias=False),
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
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing to save VRAM"""
        import torch.utils.checkpoint as checkpoint
        
        # Mark that we're using checkpointing
        self._using_checkpointing = True
        
        # Define a checkpointed forward function
        def checkpointed_forward(z):
            x = checkpoint.checkpoint(self.initial, z)
            for i, block in enumerate(self.upsample_blocks):
                x = checkpoint.checkpoint(block, x)
            x = checkpoint.checkpoint(self.penultimate, x)
            x = checkpoint.checkpoint(self.final, x)
            return x
        
        # Save original forward and replace it
        self._original_forward = self.forward
        self.forward = checkpointed_forward
        return True

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=128, features_d=64, use_spectral_norm=True):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.features_d = features_d
        self.use_spectral_norm = use_spectral_norm
        
        # Calculate number of downsampling blocks needed
        num_downsamples = int(torch.log2(torch.tensor(img_size)) - 2)
        
        # Initial block
        if use_spectral_norm:
            self.initial = nn.Sequential(
                spectral_norm(nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.initial = nn.Sequential(
                nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Downsampling blocks
        self.downsample_blocks = nn.ModuleList()
        curr_features = features_d
        
        for i in range(num_downsamples - 1):
            if use_spectral_norm:
                block = nn.Sequential(
                    spectral_norm(nn.Conv2d(curr_features, curr_features * 2, 4, 2, 1, bias=False)),
                    nn.BatchNorm2d(curr_features * 2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(curr_features, curr_features * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(curr_features * 2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.downsample_blocks.append(block)
            curr_features = curr_features * 2
        
        # Final classification block
        if use_spectral_norm:
            self.final = nn.Sequential(
                spectral_norm(nn.Conv2d(curr_features, 1, 4, 1, 0, bias=False)),
                nn.Sigmoid()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(curr_features, 1, 4, 1, 0, bias=False),
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
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing to save VRAM"""
        import torch.utils.checkpoint as checkpoint
        
        # Mark that we're using checkpointing
        self._using_checkpointing = True
        
        # Define a checkpointed forward function
        def checkpointed_forward(x):
            x = checkpoint.checkpoint(self.initial, x)
            for i, block in enumerate(self.downsample_blocks):
                x = checkpoint.checkpoint(block, x)
            x = checkpoint.checkpoint(self.final, x)
            return x.view(-1, 1).squeeze(1)
        
        # Save original forward and replace it
        self._original_forward = self.forward
        self.forward = checkpointed_forward
        return True

# These are the improved GAN modules with better gradient flow and memory efficiency
