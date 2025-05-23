import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import glob
import random
import gc  # For garbage collection
import time  # For timestamping
import argparse
import traceback  # For detailed error reporting
import subprocess
import shutil
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque

# Enable CUDA error debugging (for misaligned address and memory issues)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- MEMORY MANAGEMENT UTILITIES ---
def print_gpu_memory_status():
    """Print current GPU memory usage for debugging"""
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        print(f"GPU Memory: {used_mem/(1024**2):.1f}MB used / {total_mem/(1024**2):.1f}MB total")
        return used_mem, total_mem
    else:
        print("GPU not available")
        return 0, 0

def clean_memory():
    """Clean up memory by releasing Python objects and clearing CUDA cache"""
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return True
    return False

# --- CONFIG ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
# VRAM-friendly (default): patch training, small images
IMG_SIZE = 32
FULL_RESOLUTION = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # Reduced from 2e-4 for more stable training
WEIGHT_DECAY = 1e-5   # Add weight decay for regularization
VALIDATION_SPLIT = 0.1  # Percentage of data to use for validation
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping to prevent exploding gradients
PATIENCE = 5  # Patience for early stopping and learning rate reduction
# If not vram-friendly, use full images for both patch and full
import sys
if '--vram-friendly' not in sys.argv:
    IMG_SIZE = 128
    FULL_RESOLUTION = 128
RESOLUTION_SCALE = FULL_RESOLUTION / IMG_SIZE
GRADIENT_RES_SCALE = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility (but randomize each run)
import random
import numpy as np
import time
seed = int(time.time())
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f"Random seed for this run: {seed}")

# --- DATASET ---
class AxolotlDataset(Dataset):
    def __init__(self, folder, transform=None, augment=False):
        self.folder = folder
        self.transform = transform
        self.augment = augment
        self.reload_files()
        
    def reload_files(self):
        self.files = glob.glob(os.path.join(self.folder, '*'))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Robust to missing/corrupted files
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            
            # Apply data augmentation if enabled
            if self.augment and random.random() > 0.5:
                # Random horizontal flip
                if random.random() > 0.5:
                    img = transforms.functional.hflip(img)
                
                # Random vertical flip
                if random.random() > 0.5:
                    img = transforms.functional.vflip(img)
                
                # Random rotation (0, 90, 180, 270 degrees)
                if random.random() > 0.5:
                    angle = random.choice([0, 90, 180, 270])
                    img = transforms.functional.rotate(img, angle)
                
                # Random brightness/contrast adjustments
                if random.random() > 0.5:
                    brightness_factor = 0.8 + random.random() * 0.4  # 0.8-1.2
                    contrast_factor = 0.8 + random.random() * 0.4    # 0.8-1.2
                    img = transforms.functional.adjust_brightness(img, brightness_factor)
                    img = transforms.functional.adjust_contrast(img, contrast_factor)
                    
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: failed to load {self.files[idx]}: {e}")
            # Remove the bad file from the list
            del self.files[idx]
            # If no files left, raise
            if not self.files:
                raise RuntimeError("No valid images left in dataset!")
            # If too many files are missing, reload file list from disk
            if len(self.files) < 10:
                print("Reloading file list from disk due to too many missing files...")
                self.reload_files()
            # Clamp idx to valid range
            idx = idx % len(self.files)
            return self.__getitem__(idx)

# Data transformations with normalization
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Create datasets with augmentation for training
train_ds = AxolotlDataset(TRAIN_DIR, transform, augment=True)
test_ds = AxolotlDataset(TEST_DIR, transform, augment=False)

# Create train/validation split
total_train_size = len(train_ds)
val_size = int(total_train_size * VALIDATION_SPLIT)
train_size = total_train_size - val_size
train_subset, val_subset = torch.utils.data.random_split(
    train_ds, [train_size, val_size], 
    generator=torch.Generator().manual_seed(seed)
)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- PROGRESSIVE SCALING UNET FOR DIFFUSION ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=32, resolution_scale=1.0, dropout_rate=0.1):
        """
        Progressive scaling UNet for diffusion models
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Base feature multiplier
            resolution_scale: Resolution scaling factor (1.0 = original size, 2.0 = double resolution)
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.resolution_scale = resolution_scale
        
        # Adjust features based on resolution for better scaling
        adjusted_features = int(features * min(1.0, 1.0/resolution_scale))
        
        # Store configuration for future scaling
        self.config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'base_features': features,
            'resolution_scale': resolution_scale,
            'dropout_rate': dropout_rate
        }
        
        # Memory-efficient encoder with batch norm and residual connections
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, adjusted_features, 3, 1, 1),
            nn.BatchNorm2d(adjusted_features),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU helps with gradient flow
            nn.Conv2d(adjusted_features, adjusted_features, 3, 1, 1), 
            nn.BatchNorm2d(adjusted_features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(adjusted_features, adjusted_features*2, 3, 2, 1),
            nn.BatchNorm2d(adjusted_features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(adjusted_features*2, adjusted_features*2, 3, 1, 1),
            nn.BatchNorm2d(adjusted_features*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(adjusted_features*2, adjusted_features*4, 3, 2, 1),
            nn.BatchNorm2d(adjusted_features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(adjusted_features*4, adjusted_features*4, 3, 1, 1),
            nn.BatchNorm2d(adjusted_features*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Additional encoder layer for higher resolutions
        if resolution_scale >= 1.5:
            self.encoder4 = nn.Sequential(
                nn.Conv2d(adjusted_features*4, adjusted_features*8, 3, 2, 1),
                nn.BatchNorm2d(adjusted_features*8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adjusted_features*8, adjusted_features*8, 3, 1, 1),
                nn.BatchNorm2d(adjusted_features*8),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Memory-efficient middle block with self-attention for higher resolutions
        if resolution_scale >= 1.5:
            middle_features = adjusted_features*8
            self.middle = nn.Sequential(
                nn.Conv2d(middle_features, middle_features, 3, 1, 1),
                nn.BatchNorm2d(middle_features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(middle_features, middle_features, 3, 1, 1),
                nn.BatchNorm2d(middle_features),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            middle_features = adjusted_features*4
            self.middle = nn.Sequential(
                nn.Conv2d(middle_features, middle_features, 3, 1, 1),
                nn.BatchNorm2d(middle_features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(middle_features, middle_features, 3, 1, 1),
                nn.BatchNorm2d(middle_features),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Memory-efficient decoder with skip connections
        if resolution_scale >= 1.5:
            # Improved skip connections with proper channel concatenation
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(middle_features, adjusted_features*4, 4, 2, 1),
                nn.BatchNorm2d(adjusted_features*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate/2),  # Less dropout in decoder
                nn.Conv2d(adjusted_features*4, adjusted_features*4, 3, 1, 1),
                nn.BatchNorm2d(adjusted_features*4),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(adjusted_features*8, adjusted_features*2, 4, 2, 1),  # *8 due to skip connection
                nn.BatchNorm2d(adjusted_features*2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adjusted_features*2, adjusted_features*2, 3, 1, 1),
                nn.BatchNorm2d(adjusted_features*2),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(adjusted_features*4, adjusted_features, 4, 2, 1),  # *4 due to skip connection
                nn.BatchNorm2d(adjusted_features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adjusted_features, adjusted_features, 3, 1, 1),
                nn.BatchNorm2d(adjusted_features),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(middle_features, adjusted_features*2, 4, 2, 1),
                nn.BatchNorm2d(adjusted_features*2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate/2),
                nn.Conv2d(adjusted_features*2, adjusted_features*2, 3, 1, 1),
                nn.BatchNorm2d(adjusted_features*2),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(adjusted_features*4, adjusted_features, 4, 2, 1),  # *4 due to skip connection
                nn.BatchNorm2d(adjusted_features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adjusted_features, adjusted_features, 3, 1, 1),
                nn.BatchNorm2d(adjusted_features),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Final layer
        self.final = nn.Conv2d(adjusted_features*2, out_channels, 3, 1, 1)  # *2 due to skip connection
    
    def forward(self, x):
        # Store input shape for potential upscaling later
        input_shape = x.shape
        
        # Encoder path with skip connections
        x1 = self.encoder1(x)        # Save for skip connection
        x2 = self.encoder2(x1)       # Save for skip connection
        x3 = self.encoder3(x2)       # Save for skip connection
        
        # Additional encoding for higher resolutions
        if hasattr(self, 'encoder4'):
            x4 = self.encoder4(x3)
            # Middle block
            x_middle = self.middle(x4)
            
            # Decoder with skip connections for higher resolution
            x = self.decoder1(x_middle)
            # Add skip connection properly
            x = torch.cat([x, x3], dim=1)
            
            x = self.decoder2(x)
            # Add skip connection properly
            x = torch.cat([x, x2], dim=1)
            
            x = self.decoder3(x)
            # Add skip connection properly
            x = torch.cat([x, x1], dim=1)
        else:
            # Middle block for standard resolution
            x_middle = self.middle(x3)
            
            # Decoder with skip connections for standard resolution
            x = self.decoder1(x_middle)
            # Add skip connection properly
            x = torch.cat([x, x2], dim=1)
            
            x = self.decoder2(x)
            # Add skip connection properly
            x = torch.cat([x, x1], dim=1)
        
        # Final output layer
        x = self.final(x)
        
        return x

    def scale_model_resolution(self, new_scale):
        """
        Scale the model to a new resolution by creating a new model and transferring weights
        
        Args:
            new_scale: New resolution scale factor
            
        Returns:
            A new model with scaled resolution and transferred weights where possible
        """
        if abs(new_scale - self.resolution_scale) < 0.001:
            return self
            
        print(f"Scaling model from resolution scale {self.resolution_scale} to {new_scale}")
        
        # Create new model with new resolution scale
        new_model = SimpleUNet(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            features=self.config['base_features'],
            resolution_scale=new_scale,
            dropout_rate=self.config.get('dropout_rate', 0.1)
        )
        
        # Force both models to CPU for safer weight transfer to prevent CUDA memory issues
        original_device = next(self.parameters()).device
        
        # Ensure we're not holding any unnecessary CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Move both models to CPU for safe weight transfer
        self.to('cpu')
        new_model.to('cpu')
        
        # Additional cleanup to ensure no residual GPU memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("Transferring weights on CPU to avoid memory alignment issues")
        
        try:
            # Transfer common weights - encoder layers will always exist
            transfer_sequential_weights(self.encoder1, new_model.encoder1)
            transfer_sequential_weights(self.encoder2, new_model.encoder2)
            transfer_sequential_weights(self.encoder3, new_model.encoder3)
            
            # Transfer encoder4 if both have it
            if hasattr(self, 'encoder4') and hasattr(new_model, 'encoder4'):
                transfer_sequential_weights(self.encoder4, new_model.encoder4)
            
            # Transfer middle weights - structures may be different
            try:
                transfer_sequential_weights(self.middle, new_model.middle)
            except Exception as e:
                print(f"Could not transfer middle weights: {e}")
            
            # Transfer decoder weights - may have different structures
            try:
                transfer_sequential_weights(self.decoder1, new_model.decoder1)
                transfer_sequential_weights(self.decoder2, new_model.decoder2)
            except Exception as e:
                print(f"Could not transfer some decoder weights: {e}")
                
            # Transfer decoder3 if both have it
            if hasattr(self, 'decoder3') and hasattr(new_model, 'decoder3'):
                transfer_sequential_weights(self.decoder3, new_model.decoder3)
                
            # Transfer final layer weights if same dimensions
            try:
                transfer_layer_weights(self.final, new_model.final)
            except Exception as e:
                print(f"Could not transfer final layer weights: {e}")
            
            print("Weight transfer completed successfully")
            
            # Return new model to original device if it was on CUDA
            if original_device.type != 'cpu':
                new_model = new_model.to(original_device)
                self.to(original_device)  # Return original model to its device too
                
            return new_model
        
        except Exception as e:
            print(f"Error during model scaling: {e}")
            
            # Return original model to its original device
            if original_device.type != 'cpu':
                self.to(original_device)
                
            # If we fail, create a fresh model without weight transfer
            print("Weight transfer failed, creating new model with fresh weights")
            fresh_model = SimpleUNet(
                in_channels=self.config['in_channels'],
                out_channels=self.config['out_channels'],
                features=self.config['base_features'],
                resolution_scale=new_scale,
                dropout_rate=self.config.get('dropout_rate', 0.1)
            )
            
            # Return fresh model to original device if needed
            if original_device.type != 'cpu':
                fresh_model = fresh_model.to(original_device)
            
            return fresh_model
        
def transfer_sequential_weights(source_seq, target_seq):
    """Helper function to transfer weights between sequential layers"""
    for i, (source_layer, target_layer) in enumerate(zip(source_seq, target_seq)):
        if isinstance(source_layer, nn.Conv2d) and isinstance(target_layer, nn.Conv2d):
            try:
                # Transfer conv weights where input/output channels match
                if source_layer.weight.shape == target_layer.weight.shape:
                    target_layer.weight.data.copy_(source_layer.weight.data)
                    if source_layer.bias is not None and target_layer.bias is not None:
                        target_layer.bias.data.copy_(source_layer.bias.data)
            except Exception:
                pass
        elif isinstance(source_layer, nn.BatchNorm2d) and isinstance(target_layer, nn.BatchNorm2d):
            try:
                # Transfer batchnorm weights
                if source_layer.weight.shape == target_layer.weight.shape:
                    target_layer.weight.data.copy_(source_layer.weight.data)
                    target_layer.bias.data.copy_(source_layer.bias.data)
                    target_layer.running_mean.data.copy_(source_layer.running_mean.data)
                    target_layer.running_var.data.copy_(source_layer.running_var.data)
            except Exception:
                pass
                
def transfer_layer_weights(source_layer, target_layer):
    """Helper function to transfer weights between individual layers"""
    try:
        if source_layer.weight.shape == target_layer.weight.shape:
            target_layer.weight.data.copy_(source_layer.weight.data)
            if hasattr(source_layer, 'bias') and hasattr(target_layer, 'bias'):
                if source_layer.bias is not None and target_layer.bias is not None:
                    target_layer.bias.data.copy_(source_layer.bias.data)
    except Exception:
        pass

# --- DIFFUSION UTILS ---
def linear_beta_schedule(timesteps):
    """Linear beta schedule for diffusion process"""
    return torch.linspace(1e-4, 0.02, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule as proposed in the improved DDPM paper
    More stable for longer diffusion processes
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def forward_diffusion_sample(x_0, t, betas):
    """Apply forward diffusion to add noise to the images"""
    # Ensure noise has exactly the same shape as x_0
    noise = torch.randn(x_0.shape, device=x_0.device)
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1 - betas, dim=0))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))
    
    # Generate noisy image
    noisy_x = sqrt_alphas_cumprod[t][:, None, None, None] * x_0 + \
              sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
              
    return noisy_x, noise

def validate_model(model, val_loader, criterion, timesteps, betas, device):
    """Validate model on validation set"""
    model.eval()
    val_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for imgs in val_loader:
            # Handle patch-based vs. full image mode as in training
            if IMG_SIZE != FULL_RESOLUTION:
                if imgs.shape[2] != FULL_RESOLUTION:
                    resized_imgs = []
                    for img in imgs:
                        resized = torch.nn.functional.interpolate(
                            img.unsqueeze(0),
                            size=(FULL_RESOLUTION, FULL_RESOLUTION),
                            mode='bilinear',
                            align_corners=True
                        )
                        resized_imgs.append(resized.squeeze(0))
                    imgs = torch.stack(resized_imgs)
                
                # Use center crop for validation patches
                patches = []
                for img in imgs:
                    h, w = img.shape[1:]
                    # Center crop
                    top = (h - IMG_SIZE) // 2
                    left = (w - IMG_SIZE) // 2
                    patch = img[:, top:top+IMG_SIZE, left:left+IMG_SIZE]
                    patches.append(patch)
                imgs = torch.stack(patches)
            else:
                # Full image mode
                if imgs.shape[2] != FULL_RESOLUTION:
                    resized_imgs = []
                    for img in imgs:
                        resized = torch.nn.functional.interpolate(
                            img.unsqueeze(0),
                            size=(FULL_RESOLUTION, FULL_RESOLUTION),
                            mode='bilinear',
                            align_corners=True
                        )
                        resized_imgs.append(resized.squeeze(0))
                    imgs = torch.stack(resized_imgs)
            
            imgs = imgs.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
            
            # Apply forward diffusion
            noisy_imgs, noise = forward_diffusion_sample(imgs, t, betas)
            
            # Predict noise
            pred = model(noisy_imgs)
            
            # Ensure shapes match
            if pred.shape != noise.shape:
                target_size = (pred.shape[2], pred.shape[3])
                noise = torch.nn.functional.interpolate(
                    noise, size=target_size, mode='bilinear', align_corners=True
                )
            
            # Calculate validation loss
            loss = criterion(pred, noise)
            val_loss += loss.item()
            batch_count += 1
    
    return val_loss / batch_count if batch_count > 0 else float('inf')

def run_split_train_test(target_size):
    """Run the split_train_test.py script to recreate train/test with the current resolution."""
    split_script = os.path.join(os.path.dirname(__file__), 'split_train_test.py')
    # Remove train and test directories if they exist
    if os.path.exists(TRAIN_DIR):
        import shutil
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(TEST_DIR):
        import shutil
        shutil.rmtree(TEST_DIR)
    # Run the split script with the current patch size
    subprocess.run([
        sys.executable, split_script,
        '--size', str(target_size),
        '--train', TRAIN_DIR,
        '--test', TEST_DIR
    ], check=True)

# --- TRAINING LOOP WITH PROGRESSIVE SCALING, VALIDATION AND CHECKPOINTING ---
def train(resolution_scale=GRADIENT_RES_SCALE, continue_training=True):
    """
    Train the diffusion model using patch-wise (neural scaling) training with validation.
    
    Args:
        resolution_scale: Patch size scale for model/training (default: GRADIENT_RES_SCALE)
        continue_training: Whether to continue from checkpoint
    """
    full_img_size = FULL_RESOLUTION
    patch_size = IMG_SIZE
    print(f"Training with patch size: {patch_size}x{patch_size}, full image size: {full_img_size}x{full_img_size}")
    
    # Create model with dropout for regularization
    model = SimpleUNet(resolution_scale=RESOLUTION_SCALE, dropout_rate=0.1).to(DEVICE)
    
    # Enable gradient checkpointing if available (reduces memory usage at cost of increased computation)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Optimizer with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler to reduce LR on plateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=PATIENCE//2,
        verbose=True,
        min_lr=1e-6
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Diffusion parameters - using cosine schedule for better stability
    timesteps = 500
    betas = cosine_beta_schedule(timesteps).to(DEVICE)
    
    # Mixed precision training if available
    scaler = torch.amp.GradScaler() if DEVICE.type == 'cuda' else None
    
    # Training state
    start_epoch = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    last_few_losses = deque(maxlen=100)  # Track more losses for better averaging
    no_improve_count = 0  # Count epochs with no improvement for early stopping
    
    # Overfitting detection variables
    overfit_detection_window = 50  # Require 50 consecutive detections
    overfit_check_interval = 1000  # Check every 1000 epochs
    overfit_counter = 0
    overfit_last_epoch = 0
    
    # Checkpoint path
    checkpoint_path = os.path.join(DATA_DIR, 'diffusion_checkpoint.pth')
    
    # Resume from checkpoint if available
    if continue_training and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', best_loss)
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            no_improve_count = checkpoint.get('no_improve_count', 0)
            
            if 'scheduler' in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                except:
                    print("Could not load scheduler state")
            
            print(f"Resumed from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}, starting fresh")
    
    def safe_cleanup():
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Main training loop
    while True:
        # Training phase
        model.train()
        running_loss = 0
        batch_count = 0
        safe_cleanup()
        
        for batch_idx, imgs in enumerate(train_loader):
            retry_count = 0
            while retry_count < 3:
                try:
                    # VRAM-friendly: patch training, else full image
                    if IMG_SIZE != FULL_RESOLUTION:
                        # Always resize images to full resolution
                        if imgs.shape[2] != FULL_RESOLUTION:
                            resized_imgs = []
                            for img in imgs:
                                resized = torch.nn.functional.interpolate(
                                    img.unsqueeze(0),
                                    size=(FULL_RESOLUTION, FULL_RESOLUTION),
                                    mode='bilinear',
                                    align_corners=True
                                )
                                resized_imgs.append(resized.squeeze(0))
                            imgs = torch.stack(resized_imgs)
                        # Random crop patch for each image in the batch
                        patches = []
                        for img in imgs:
                            h, w = img.shape[1:]
                            top = torch.randint(0, h - IMG_SIZE + 1, (1,)).item() if h > IMG_SIZE else 0
                            left = torch.randint(0, w - IMG_SIZE + 1, (1,)).item() if w > IMG_SIZE else 0
                            patch = img[:, top:top+IMG_SIZE, left:left+IMG_SIZE]
                            patches.append(patch)
                        imgs = torch.stack(patches)
                    else:
                        # Full image mode: just ensure correct size
                        if imgs.shape[2] != FULL_RESOLUTION:
                            resized_imgs = []
                            for img in imgs:
                                resized = torch.nn.functional.interpolate(
                                    img.unsqueeze(0),
                                    size=(FULL_RESOLUTION, FULL_RESOLUTION),
                                    mode='bilinear',
                                    align_corners=True
                                )
                                resized_imgs.append(resized.squeeze(0))
                            imgs = torch.stack(resized_imgs)
                    
                    imgs = imgs.to(DEVICE)
                    t = torch.randint(0, timesteps, (imgs.size(0),), device=DEVICE).long()
                    noisy_imgs, noise = forward_diffusion_sample(imgs, t, betas)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            pred = model(noisy_imgs)
                            if pred.shape != noise.shape:
                                target_size = (pred.shape[2], pred.shape[3])
                                noise = torch.nn.functional.interpolate(
                                    noise,
                                    size=target_size,
                                    mode='bilinear',
                                    align_corners=True
                                )
                            assert pred.shape == noise.shape, f"Shape mismatch: pred {pred.shape} vs noise {noise.shape}"
                            loss = criterion(pred, noise)
                            
                            # Scale loss and backpropagate
                            scaler.scale(loss).backward()
                            
                            # Apply gradient clipping
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                            
                            # Update weights
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        pred = model(noisy_imgs)
                        if pred.shape != noise.shape:
                            target_size = (pred.shape[2], pred.shape[3])
                            noise = torch.nn.functional.interpolate(
                                noise,
                                size=target_size,
                                mode='bilinear',
                                align_corners=True
                            )
                        assert pred.shape == noise.shape, f"Shape mismatch: pred {pred.shape} vs noise {noise.shape}"
                        loss = criterion(pred, noise)
                        loss.backward()
                        
                        # Apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                        
                        optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Epoch {start_epoch+1} - Batch {batch_idx+1}/{len(train_loader)}, "
                              f"Loss: {running_loss/batch_count:.4f}, Patch: {patch_size}, Full: {full_img_size}")
                        if DEVICE.type == 'cuda':
                            torch.cuda.empty_cache()
                    break
                except RuntimeError as e:
                    if ("CUDNN_STATUS_EXECUTION_FAILED" in str(e)) or ("CUDA out of memory" in str(e)):
                        print(f"cuDNN/CUDA error detected on batch {batch_idx}, retry {retry_count+1}/3. Attempting recovery...")
                        clean_memory()
                        torch.cuda.empty_cache()
                        retry_count += 1
                        time.sleep(2)
                        continue
                    else:
                        raise
                except Exception as e:
                    print(f"Non-CUDA error on batch {batch_idx}: {e}")
                    raise
        
        # Calculate average training loss
        avg_loss = running_loss / len(train_loader)
        last_few_losses.append(avg_loss)
        
        # Validation phase
        val_loss = validate_model(model, val_loader, criterion, timesteps, betas, DEVICE)
        print(f"Epoch {start_epoch+1} completed - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            try:
                torch.save({
                    'state_dict': model.state_dict(),
                    'resolution_scale': RESOLUTION_SCALE,
                    'config': model.config,
                    'val_loss': best_val_loss,
                    'epoch': start_epoch,
                    'betas': betas
                }, os.path.join(DATA_DIR, 'best_diffusion_model.pth'))
                print(f"✓ New best model saved with val loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Failed to save best model: {e}")
        else:
            no_improve_count += 1
            print(f"No improvement in validation loss for {no_improve_count} epochs.")
        
        # Early stopping
        if no_improve_count >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without improvement.")
            break
        
        # Overfitting detection (check every 1000 epochs)
        if (start_epoch + 1) % overfit_check_interval == 0:
            # Example overfitting condition: val_loss increases while train_loss decreases
            if len(last_few_losses) >= 10:
                recent_train = sum(list(last_few_losses)[-10:]) / 10
                if val_loss > best_val_loss * 1.2 and recent_train < best_loss * 0.8:
                    overfit_counter += 1
                    print(f"[Overfitting detection] {overfit_counter}/{overfit_detection_window} possible overfitting events.")
                else:
                    overfit_counter = 0
            if overfit_counter >= overfit_detection_window:
                print(f"Stopping training due to detected overfitting ({overfit_counter} consecutive detections).")
                break
        
        # Save model checkpoint periodically
        if (start_epoch + 1) % 5 == 0:
            try:
                # Checkpoint includes state for resuming training
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'epoch': start_epoch,
                    'best_loss': avg_loss,
                    'best_val_loss': best_val_loss,
                    'resolution_scale': RESOLUTION_SCALE,
                    'no_improve_count': no_improve_count
                }, checkpoint_path)
                
                # Also save a named checkpoint with epoch number
                ckpt_file = os.path.join(DATA_DIR, f'diffusion_model_epoch{start_epoch+1}_res{RESOLUTION_SCALE}.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                    'resolution_scale': RESOLUTION_SCALE,
                    'config': model.config,
                    'val_loss': val_loss,
                    'train_loss': avg_loss,
                    'betas': betas
                }, ckpt_file)
                
                print(f"✓ Saved checkpoint at epoch {start_epoch+1}")
                
                # Keep only recent checkpoints to save disk space
                ckpts = sorted(glob.glob(os.path.join(DATA_DIR, 'diffusion_model_epoch*.pth')), 
                               key=os.path.getmtime, reverse=True)
                for old_ckpt in ckpts[3:]:
                    try:
                        os.remove(old_ckpt)
                    except Exception as e:
                        print(f"Failed to remove old checkpoint {old_ckpt}: {e}")
            except Exception as e:
                print(f"Error during checkpoint saving: {e}")
        
        # Generate a sample image periodically to visualize progress
        if (start_epoch+1) % 25 == 0 or (start_epoch+1) % 100 == 0:
            print(f"Generating sample image at epoch {start_epoch+1}...")
            try:
                sample_image(
                    model=model,  # Pass model directly to avoid loading from disk
                    out_path=os.path.join(DATA_DIR, f'sample_epoch.png'),
                    steps=250,
                    resolution_scale=RESOLUTION_SCALE,
                    betas=betas,
                    upscale_factor=1.0
                )
            except Exception as e:
                print(f"Failed to generate sample image: {e}")
                traceback.print_exc()
        
        safe_cleanup()
        start_epoch += 1
        
        # Check if we've trained for enough epochs
        max_epochs = 10000  # Set a reasonable maximum
        if start_epoch >= max_epochs:
            print(f"Reached maximum epochs ({max_epochs}). Training complete.")
            break
    
    print("Training complete.")
    
    # Generate final sample
    try:
        print("Generating final sample image...")
        sample_image(
            model=model,
            out_path=os.path.join(DATA_DIR, f'final_sample.png'),
            steps=500,  # More steps for final sample
            resolution_scale=RESOLUTION_SCALE,
            betas=betas,
            upscale_factor=1.5  # Higher quality final sample
        )
    except Exception as e:
        print(f"Failed to generate final sample: {e}")
        
    return model, best_val_loss

# --- IMAGE SAMPLING FUNCTION WITH RESOLUTION SUPPORT ---
def sample_image(model_path=None, out_path=None, steps=500, use_cpu_fallback=False, 
                resolution_scale=None, upscale_factor=1.0):
    """
    Sample an image from the diffusion model with resolution control.
    Ensures output is a square image and starts at a reasonable low resolution (default 64x64).
    """
    # Clear GPU memory before sampling
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Determine device for sampling
    sampling_device = torch.device('cpu') if use_cpu_fallback else DEVICE
    print(f"Sampling using device: {sampling_device}")
    
    try:
        # Load model with proper error handling
        if model_path is None:
            model_path = os.path.join(DATA_DIR, 'best_diffusion_model.pth')
            
        # Check if the file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        
        # Load model with resolution information
        model_data = torch.load(model_path, map_location='cpu')
        
        # Handle different saved formats (backward compatibility)
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            # New format with metadata
            model_state = model_data['state_dict']
            native_resolution = model_data.get('resolution_scale', 1.0)
            config = model_data.get('config', {'resolution_scale': native_resolution})
        else:
            # Old format (just state dict)
            model_state = model_data
            native_resolution = 1.0
            config = {'resolution_scale': native_resolution}
        
        # Set a reasonable low-res default for sampling
        default_sample_size = 64
        base_img_size = default_sample_size
        
        # Allow override by resolution_scale
        if resolution_scale is not None:
            final_resolution = resolution_scale
        else:
            final_resolution = 1.0
        
        actual_img_size = int(base_img_size * final_resolution)
        
        # Ensure square image
        actual_img_size = max(8, int(round(actual_img_size)))
        print(f"Generating square image at {actual_img_size}x{actual_img_size}")
        
        # Create model with appropriate resolution
        model = SimpleUNet(resolution_scale=final_resolution)
        
        # If we need to change resolution from the native model resolution
        if final_resolution != native_resolution and abs(final_resolution - native_resolution) > 0.01:
            print(f"Scaling model from {native_resolution} to {final_resolution}")
            
            # Create temp model with original resolution and load state
            temp_model = SimpleUNet(resolution_scale=native_resolution)
            temp_model.load_state_dict(model_state)
            
            # Scale to target resolution
            model = temp_model.scale_model_resolution(final_resolution)
        else:
            # Load directly
            model.load_state_dict(model_state)
        
        # Move to appropriate device
        model = model.to(sampling_device)
        model.eval()
        
        # Generate initial random noise on CPU then move to device
        img = torch.randn(1, 3, actual_img_size, actual_img_size, device='cpu').to(sampling_device)
        
        # TODO: Replace this with the actual diffusion denoising loop for real samples
        out_img = img
        if upscale_factor != 1.0:
            out_img = torch.nn.functional.interpolate(out_img, scale_factor=upscale_factor, mode='bilinear', align_corners=True)
        out_img = (out_img.clamp(-1, 1) + 1) / 2  # Denormalize to [0,1]
        save_image(out_img, out_path)
        print(f"Sample image saved to {out_path}")
    except Exception as e:
        print(f"Error during sample generation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train or sample from the diffusion model.")
    parser.add_argument('--sample-now', action='store_true', help='Generate a sample image immediately using the best model.')
    parser.add_argument('--sample-steps', type=int, default=250, help='Number of diffusion steps for sampling.')
    parser.add_argument('--sample-output', type=str, default=None, help='Output path for the sample image.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU for sampling.')
    parser.add_argument('--resolution-scale', type=float, default=None, help='Override model resolution scale for sampling.')
    parser.add_argument('--upscale', type=float, default=1.0, help='Upscale factor for sample image.')
    parser.add_argument('--train', action='store_true', help='Train the diffusion model.')
    parser.add_argument('--fresh', action='store_true', help='Start training from scratch (do not resume).')
    args = parser.parse_args()

    if args.sample_now:
        print("Generating sample image using the best model...")
        out_path = args.sample_output or os.path.join(DATA_DIR, 'sample_now.png')
        sample_image(
            model_path=os.path.join(DATA_DIR, 'best_diffusion_model.pth'),
            out_path=out_path,
            steps=args.sample_steps,
            use_cpu_fallback=args.cpu,
            resolution_scale=args.resolution_scale,
            upscale_factor=args.upscale
        )
        print(f"Sample image saved to {out_path}")
        return

    if args.train:
        print("Starting training...")
        clean_memory()
        train(
            resolution_scale=args.resolution_scale or GRADIENT_RES_SCALE,
            continue_training=not args.fresh
        )
        return

    print("No action specified. Use --train to train or --sample-now to generate a sample image.")

if __name__ == "__main__":
    main()
