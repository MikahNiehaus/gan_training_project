import os
import time
import glob
import yaml
import torch
import random
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from models.gan_modules import Generator, Discriminator
from models.gan_weight_transfer import transfer_gan_weights

class GANConfig:
    """Configuration manager for GAN training"""
    
    def __init__(self, config_path='config.yaml'):
        """Load configuration from YAML file"""
        self.config_path = config_path
        self.reload()
        
    def reload(self):
        """Reload configuration from file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set all config properties as attributes
        for key, value in config.items():
            setattr(self, key, value)
            
        # Auto-detect device if set to 'auto'
        if self.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)
            
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        return self

class AxolotlDataset(Dataset):
    """Dataset handler for loading and preprocessing images"""
    
    def __init__(self, folder, transform=None, preload=False, cache_tensors=False):
        self.folder = folder
        self.transform = transform
        self.preload = preload
        self.cache_tensors = cache_tensors
        self.reload_files()
        self.images = None
        self.tensors = None
        
        if self.preload:
            print(f"[DATA] Preloading {len(self.files)} images into RAM...")
            self.images = []
            for f in self.files:
                try:
                    img = Image.open(f).convert('RGB')
                    self.images.append(img)
                except Exception as e:
                    print(f"[DATA] Failed to preload {f}: {e}")
            print(f"[DATA] Preloaded {len(self.images)} images into RAM.")
            
            if self.cache_tensors:
                print(f"[DATA] Caching all transformed tensors in RAM...")
                self.tensors = []
                for img in self.images:
                    try:
                        tensor = self.transform(img) if self.transform else img
                        self.tensors.append(tensor)
                    except Exception as e:
                        print(f"[DATA] Failed to transform image for tensor cache: {e}")
                print(f"[DATA] Cached {len(self.tensors)} tensors in RAM.")

    def reload_files(self):
        """Reload image file list from directory"""
        self.files = glob.glob(os.path.join(self.folder, '*'))
        if len(self.files) == 0:
            raise RuntimeError(f"No files found in {self.folder}")

    def __len__(self):
        if self.cache_tensors and self.tensors is not None:
            return len(self.tensors)
        return len(self.files) if not self.preload else len(self.images)

    def __getitem__(self, idx):
        try:
            if self.cache_tensors and self.tensors is not None:
                return self.tensors[idx]
            if self.preload and self.images is not None:
                img = self.images[idx]
            else:
                img = Image.open(self.files[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: failed to load {self.files[idx]}: {e}")
            if not self.preload:
                # Handle bad files by removing them from the list
                del self.files[idx]
                if not self.files:
                    raise RuntimeError("No valid images left in dataset!")
                if len(self.files) < 10:
                    print("Reloading file list from disk due to too many missing files...")
                    self.reload_files()
                idx = idx % len(self.files)
                return self.__getitem__(idx)
            else:
                raise e

class AugmentedDataset(IterableDataset):
    """IterableDataset that yields randomly augmented samples"""
    
    def __init__(self, folder, transform=None, target_size=20000):
        self.folder = folder
        self.transform = transform
        self.target_size = target_size
        self.files = glob.glob(os.path.join(self.folder, '*'))
        if len(self.files) == 0:
            raise RuntimeError(f"No files found in {self.folder}")

    def __iter__(self):
        for _ in range(self.target_size):
            img_path = random.choice(self.files)
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                yield img
            except Exception as e:
                continue  # Skip failed images

class GANTrainer:
    """GAN Trainer with progressive growing and optimization features"""
    
    def __init__(self, config):
        """Initialize the GAN trainer using the provided configuration"""
        self.config = config
        self.setup_seeds()
        self.setup_models()
        self.setup_training_state()
        
    def setup_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            
    def setup_models(self):
        """Initialize models, optimizers, and criterion"""
        self.img_size = self.config.resolutions[0]  # Start with the lowest resolution
        self.device = self.config.device
        
        # Create models
        self.G = Generator(
            z_dim=self.config.z_dim, 
            img_channels=self.config.img_channels, 
            img_size=self.img_size
        ).to(self.device)
        
        self.D = Discriminator(
            img_channels=self.config.img_channels, 
            img_size=self.img_size
        ).to(self.device)
        
        # Create optimizers
        self.opt_G = optim.Adam(
            self.G.parameters(), 
            lr=self.config.learning_rate_g, 
            betas=tuple(self.config.adam_betas)
        )
        
        self.opt_D = optim.Adam(
            self.D.parameters(), 
            lr=self.config.learning_rate_d, 
            betas=tuple(self.config.adam_betas)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def setup_training_state(self):
        """Initialize training state variables"""
        self.start_epoch = 0
        self.epochs_at_res = 0
        self.fixed_noise = torch.randn(16, self.config.z_dim, 1, 1, device=self.device)
        self.checkpointing_level = self.config.initial_checkpointing_level
        self.resolution_history = []
        
        # VRAM optimization settings from config
        vram_optimize = getattr(self.config, 'vram_optimization', {})
        self.max_checkpointing_level = vram_optimize.get('max_checkpointing_level', 3)
        self.max_vram_retries = vram_optimize.get('max_vram_retries', 3)
        self.auto_reduce_batch_size = vram_optimize.get('auto_reduce_batch_size', True)
        self.auto_reduce_learning_rates = vram_optimize.get('auto_reduce_learning_rates', True)
        
        # For pixel-perfect matching
        self.reference_image_tensor = None
        self.fixed_seed_noise = self._get_fixed_seed_noise()
        
        # Load reference image if needed
        if hasattr(self.config, 'reference_image_path') and self.config.reference_image_path:
            self.reference_image_tensor = self._load_reference_image(self.config.reference_image_path)
        
    def _get_fixed_seed_noise(self):
        """Generate fixed noise for consistent samples"""
        g = torch.Generator(device=self.device)
        g.manual_seed(self.config.fixed_seed)
        return torch.randn(1, self.config.z_dim, 1, 1, device=self.device, generator=g)
        
    def _load_reference_image(self, path):
        """Load and preprocess the reference image to tensor"""
        img = Image.open(path).convert('RGB')
        transform = self.get_transform(self.img_size, augment=False)
        return transform(img).unsqueeze(0)  # Add batch dim
        
    def _images_equal(self, img1, img2):
        """Check if two image tensors are pixel-identical"""
        def to_uint8(t):
            t = t.detach().cpu()
            t = (t * 0.5 + 0.5).clamp(0, 1) * 255
            return t.round().to(torch.uint8)
        return torch.equal(to_uint8(img1), to_uint8(img2))
        
    def _save_pixelmatch_outputs(self, epoch, gen_img, real_img):
        """Save outputs when a pixel-perfect match is found"""
        out_dir = os.path.join(self.config.sample_dir, 'pixelmatch_success')
        os.makedirs(out_dir, exist_ok=True)
        
        # Save seed
        with open(os.path.join(out_dir, 'fixed_seed.txt'), 'w') as f:
            f.write(str(self.config.fixed_seed))
            
        # Save model
        self.save_full_model(epoch, manual=True)
        
        # Save images
        save_image(gen_img, os.path.join(out_dir, 'generated.png'), normalize=True)
        save_image(real_img, os.path.join(out_dir, 'reference.png'), normalize=True)
        print(f"[SUCCESS] Pixel-perfect match found at epoch {epoch+1}. Outputs saved to {out_dir}")
        
    def get_transform(self, img_size, augment=True):
        """Get image transformation pipeline"""
        if not augment:
            # No augmentation: just resize, center crop, tensor, normalize
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            
        # Data augmentation: H flip, brightness/contrast, center crop/pad
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.CenterCrop(img_size),
            transforms.Pad(padding=4, fill=0, padding_mode='constant'),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
    def create_dataloader(self, resolution):
        """Create a data loader for the current resolution"""
        transform = self.get_transform(
            resolution, 
            augment=self.config.augment_data
        )
        
        # Determine batch size based on resolution
        batch_size = self.config.batch_size_strategy.get(
            resolution, 
            self.config.base_batch_size
        )
        
        # Create dataset
        if self.config.augment_data:
            dataset = AugmentedDataset(
                self.config.train_dir,
                transform,
                target_size=self.config.augment_target_size
            )
            shuffle = False  # IterableDataset doesn't support shuffle
        else:
            dataset = AxolotlDataset(
                self.config.train_dir,
                transform,
                preload=self.config.preload_to_ram,
                cache_tensors=self.config.cache_tensors
            )
            shuffle = True
            
        # Create DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.config.dataloader_workers,
            pin_memory=self.config.dataloader_pin_memory,
            persistent_workers=self.config.dataloader_persistent_workers
        )
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM"""
        try:
            from models.checkpoint_utils import enable_gradient_checkpointing
            
            enable_gradient_checkpointing(self.G, level=self.checkpointing_level)
            print(f"[VRAM] Enabled gradient checkpointing on Generator (level {self.checkpointing_level})")
            
            enable_gradient_checkpointing(self.D, level=self.checkpointing_level)
            print(f"[VRAM] Enabled gradient checkpointing on Discriminator (level {self.checkpointing_level})")
            
            return True
        except Exception as e:
            print(f"[VRAM] Error enabling gradient checkpointing: {str(e)}")
            return False
            
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        try:
            from models.checkpoint_utils import disable_gradient_checkpointing
            
            disable_gradient_checkpointing(self.G)
            disable_gradient_checkpointing(self.D)
            
            print("[VRAM] Disabled gradient checkpointing")
            return True
        except Exception as e:
            print(f"[VRAM] Error disabling gradient checkpointing: {str(e)}")
            return False
            
    def grow_to_resolution(self, new_img_size):
        """Grow the current model to a new resolution, copying over weights in-place"""
        if new_img_size == self.img_size:
            print(f"[GROW] Model already at {new_img_size}x{new_img_size}.")
            return
            
        print(f"[GROW] Growing model from {self.img_size}x{self.img_size} to {new_img_size}x{new_img_size}...")
        
        # Create new models at the higher resolution
        new_G = Generator(
            z_dim=self.config.z_dim, 
            img_channels=self.config.img_channels, 
            img_size=new_img_size
        ).to(self.device)
        
        new_D = Discriminator(
            img_channels=self.config.img_channels, 
            img_size=new_img_size
        ).to(self.device)
        
        # Transfer weights from old to new
        transfer_gan_weights(self.G, new_G)
        transfer_gan_weights(self.D, new_D)
        
        # Replace models in-place
        self.G = new_G
        self.D = new_D
        self.img_size = new_img_size
        
        print(f"[GROW] Model grown to {new_img_size}x{new_img_size} and weights transferred.")
        
        # Re-create optimizers for new parameters
        self.opt_G = torch.optim.Adam(
            self.G.parameters(), 
            lr=self.config.learning_rate_g, 
            betas=tuple(self.config.adam_betas)
        )
        
        self.opt_D = torch.optim.Adam(
            self.D.parameters(), 
            lr=self.config.learning_rate_d, 
            betas=tuple(self.config.adam_betas)
        )
        
        # Reset fixed seed noise for new resolution
        self.fixed_seed_noise = self._get_fixed_seed_noise()
        
        # Update reference image if exists
        if self.reference_image_tensor is not None and hasattr(self.config, 'reference_image_path'):
            self.reference_image_tensor = self._load_reference_image(self.config.reference_image_path)
            
    def load_checkpoint(self, resolution=None):
        """Load model checkpoint"""
        checkpoint_path = self.config.checkpoint_path
        
        # If resolution is specified, look for resolution-specific checkpoint
        if resolution is not None:
            resolution_ckpt = os.path.join(
                os.path.dirname(checkpoint_path),
                f'gan_checkpoint_{resolution}.pth'
            )
            if os.path.exists(resolution_ckpt):
                checkpoint_path = resolution_ckpt
                print(f"[CHECKPOINT] Found resolution-specific checkpoint for {resolution}x{resolution}")
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Check resolution match
                checkpoint_img_size = checkpoint.get('img_size', self.img_size)
                
                # If checkpoint is for a different resolution and we're not trying to load a specific resolution
                if checkpoint_img_size != self.img_size and resolution is None:
                    print(f"[WARN] Resolution mismatch: Checkpoint is for {checkpoint_img_size}x{checkpoint_img_size}, but current model is {self.img_size}x{self.img_size}. Starting from scratch.")
                    return
                
                # Ensure models match checkpoint resolution
                if checkpoint_img_size != self.img_size:
                    self.G = Generator(
                        z_dim=self.config.z_dim, 
                        img_channels=self.config.img_channels, 
                        img_size=checkpoint_img_size
                    ).to(self.device)
                    
                    self.D = Discriminator(
                        img_channels=self.config.img_channels, 
                        img_size=checkpoint_img_size
                    ).to(self.device)
                    
                    self.img_size = checkpoint_img_size
                
                # Load model weights
                self.G.load_state_dict(checkpoint['G'])
                self.D.load_state_dict(checkpoint['D'])
                
                # Create new optimizers and load their states
                self.opt_G = optim.Adam(
                    self.G.parameters(), 
                    lr=self.config.learning_rate_g, 
                    betas=tuple(self.config.adam_betas)
                )
                
                self.opt_D = optim.Adam(
                    self.D.parameters(), 
                    lr=self.config.learning_rate_d, 
                    betas=tuple(self.config.adam_betas)
                )
                
                self.opt_G.load_state_dict(checkpoint['opt_G'])
                self.opt_D.load_state_dict(checkpoint['opt_D'])
                
                # Make sure we're not using gradient checkpointing when loading
                self.disable_gradient_checkpointing()
                
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"[INFO] Successfully loaded checkpoint from {checkpoint_path} at epoch {self.start_epoch}")
                
                return True
            except Exception as e:
                print(f"[WARN] Error loading checkpoint: {str(e)}. Starting from scratch.")            return False
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
            return False
            
    def save_checkpoint(self, epoch, resolution=None):
        """Save model checkpoint"""
        # Get clean model states first
        self.disable_gradient_checkpointing()
        
        # Always use resolution in checkpoint filename
        curr_resolution = resolution if resolution is not None else self.img_size
        checkpoint_dir = os.path.dirname(self.config.checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, f'gan_checkpoint_{curr_resolution}.pth')
        
        # Use a temporary checkpoint path
        temp_checkpoint_path = checkpoint_path + ".tmp"
        
        try:
            # Save the checkpoint to temporary file first
            torch.save({
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'opt_G': self.opt_G.state_dict(),
                'opt_D': self.opt_D.state_dict(),
                'epoch': epoch,
                'img_size': self.img_size  # Store current image size
            }, temp_checkpoint_path)
            
            # Rename temp file to actual checkpoint path
            if os.path.exists(temp_checkpoint_path):
                import shutil
                shutil.move(temp_checkpoint_path, checkpoint_path)
                
            print(f"[Checkpoint] Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
            
            # Log resolution history
            self.resolution_history.append({'epoch': epoch+1, 'img_size': self.img_size})
            
            # Save epoch-specific checkpoint if needed
            if (epoch + 1) % self.config.save_epoch_checkpoint_interval == 0:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                epoch_checkpoint_path = os.path.join(checkpoint_dir, f'gan_checkpoint_{curr_resolution}_epoch{epoch+1}.pth')
                
                # Copy the checkpoint file (faster than saving again)
                import shutil
                shutil.copy(checkpoint_path, epoch_checkpoint_path)
                print(f"[CHECKPOINT] Saved epoch checkpoint: {epoch_checkpoint_path}")
                
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {str(e)}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)
            return False
              def save_full_model(self, epoch, manual=False):
        """Save the full generator model with metadata"""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.config.full_model_path), exist_ok=True)
        
        # Use resolution in the filename
        model_dir = os.path.dirname(self.config.full_model_path)
        resolution_model_path = os.path.join(model_dir, f'gan_full_model_{self.img_size}.pth')
        
        # Absolute path for logging
        full_model_abs_path = os.path.abspath(resolution_model_path)
        print(f"[FullModel] Saving full model to: {full_model_abs_path}")
        
        # Save the full model (generator only, plus metadata)
        torch.save({
            'G': self.G.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'epoch': epoch,
            'img_size': self.img_size,
            'resolution_history': self.resolution_history
        }, full_model_abs_path)
        
        # Verify the file was created
        if os.path.exists(full_model_abs_path):
            file_size = os.path.getsize(full_model_abs_path)
            print(f"[FullModel] Saved full model at epoch {epoch+1} to {full_model_abs_path} (Size: {file_size} bytes)")
            
        # Save epoch-specific full model if needed (with resolution in name)
        if (epoch + 1) % self.config.save_epoch_checkpoint_interval == 0 or manual:
            epoch_model_path = os.path.join(model_dir, f'gan_full_model_{self.img_size}_epoch{epoch+1}.pth')
            
            # Copy the model file (faster than saving again)
            import shutil
            shutil.copy(full_model_abs_path, epoch_model_path)
            print(f"[FULLMODEL] Saved epoch full model: {epoch_model_path}")
            
    def generate_sample(self, epoch):
        """Generate sample images from the current model"""
        try:
            # Set eval mode and disable checkpointing
            self.G.eval()
            self.disable_gradient_checkpointing()

            # Generate single image first (smaller memory footprint)
            with torch.no_grad():
                fake = self.G(self.fixed_noise[:1]).detach().cpu()
                save_image(fake, os.path.join(self.config.sample_dir, f'sample_epoch.png'), normalize=True)

                # Generate a grid of samples
                try:
                    torch.cuda.empty_cache()  # Clear memory first
                    # Use truly new random noise for each image in the grid
                    grid_noise = torch.randn(16, self.config.z_dim, 1, 1, device=self.device)
                    fake_grid = self.G(grid_noise).detach().cpu()
                    save_image(
                        fake_grid, 
                        os.path.join(self.config.sample_dir, 'sample_grid.png'), 
                        normalize=True, 
                        nrow=4, 
                        padding=2, 
                        pad_value=1
                    )
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print("[VRAM] Grid sample generation skipped due to memory constraints")
                    else:
                        print(f"[ERROR] Grid sample generation error: {str(e)}")
                        
                # Save epoch-specific grid at intervals
                if (epoch + 1) % self.config.save_epoch_checkpoint_interval == 0:
                    try:
                        grid_path = os.path.join(self.config.sample_dir, f'sample_grid_epoch{epoch+1}.png')
                        save_image(
                            fake_grid, 
                            grid_path, 
                            normalize=True, 
                            nrow=4, 
                            padding=2, 
                            pad_value=1
                        )
                        print(f"[SAMPLE] Saved grid image: {grid_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to save grid image for epoch {epoch+1}: {e}")
        except Exception as e:
            print(f"[ERROR] Sample generation failed: {str(e)}")
            # Fall back to CPU if necessary
            try:
                print("[VRAM] Attempting to generate sample on CPU...")
                self.G = self.G.cpu()
                with torch.no_grad():
                    fake = self.G(self.fixed_noise[:1].cpu()).detach()
                    save_image(fake, os.path.join(self.config.sample_dir, 'sample_grid.png'), normalize=True)
                self.G = self.G.to(self.device)
            except Exception as e2:
                print(f"[CRITICAL] CPU fallback failed too: {str(e2)}")
        finally:
            # Restore model to training state
            self.G.train()
            # Re-enable gradient checkpointing if needed
            if self.config.enable_gradient_checkpointing and self.checkpointing_level > 0:
                self.enable_gradient_checkpointing()
                
    def train(self, epochs=None, resolution=None):
        """Train the GAN for the specified epochs at the given resolution"""
        if resolution is not None:
            # Update model resolution if needed
            if resolution != self.img_size:
                self.grow_to_resolution(resolution)
        
        # Create data loader for training
        train_loader = self.create_dataloader(self.img_size)
        
        # Try to load checkpoint
        self.load_checkpoint()
        
        # Get epoch count
        if epochs is None:
            # Use minimum epochs from config
            epochs = self.config.min_epochs_per_res.get(self.img_size, 100)
        
        # Initialize training variables
        epoch = self.start_epoch
        max_epochs = epoch + epochs
        distortion = self.config.manual_distortion_scale if self.config.use_distortion else 0.0
        vram_retry = 0
        avg_D_loss = None
        avg_G_loss = None
        
        # Enable gradient checkpointing if configured
        if self.config.enable_gradient_checkpointing:
            self.enable_gradient_checkpointing()
            
        # Reduce learning rates for high resolutions
        if self.auto_reduce_learning_rates and self.img_size >= 512:
            print(f"[VRAM] High resolution detected ({self.img_size}x{self.img_size}). Auto-reducing learning rates.")
            for param_group in self.opt_G.param_groups:
                param_group['lr'] = self.config.learning_rate_g * 0.5
            for param_group in self.opt_D.param_groups:
                param_group['lr'] = self.config.learning_rate_d * 0.5
        
        if self.auto_reduce_learning_rates and self.img_size >= 720:
            print(f"[VRAM] Very high resolution detected ({self.img_size}x{self.img_size}). Auto-reducing learning rates further.")
            for param_group in self.opt_G.param_groups:
                param_group['lr'] = self.config.learning_rate_g * 0.25
            for param_group in self.opt_D.param_groups:
                param_group['lr'] = self.config.learning_rate_d * 0.25
                
        # Main training loop
        while epoch < max_epochs:
            # Set a new random seed for each epoch
            epoch_seed = int(time.time()) + epoch
            random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            np.random.seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(epoch_seed)
                
            try:
                # Tracking variables for this epoch
                D_losses = []
                G_losses = []
                D_paused_batches = 0
                G_paused_batches = 0
                epoch_start = time.time()
                
                # Create progress bar
                pbar = None
                if self.config.show_progress_bar:
                    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [{self.img_size}x{self.img_size}]", leave=False)
                    data_iter = pbar
                else:
                    data_iter = train_loader
                    
                # Batch timing
                batch_times = []
                data_times = []
                batch_start = time.time()
                
                # Process each batch
                for real in data_iter:
                    data_loaded = time.time()
                    data_times.append(data_loaded - batch_start)
                    
                    real = real.to(self.device)
                    batch_size = real.size(0)
                    noise = torch.randn(batch_size, self.config.z_dim, 1, 1, device=self.device)
                    
                    # Make sure we're using the configured learning rates
                    for param_group in self.opt_G.param_groups:
                        param_group['lr'] = self.config.learning_rate_g
                    for param_group in self.opt_D.param_groups:
                        param_group['lr'] = self.config.learning_rate_d
                        
                    # TRAIN DISCRIMINATOR
                    # Only update discriminator if it's not too strong
                    train_discriminator = True
                    if epoch > 0 and avg_D_loss is not None and avg_G_loss is not None:
                        if avg_D_loss < 0.3 and avg_D_loss < avg_G_loss / 2:
                            # If discriminator is too strong, skip training it this batch
                            train_discriminator = random.random() > 0.5
                            
                    if train_discriminator:
                        self.opt_D.zero_grad()
                        
                        # Generate fake images without gradients
                        with torch.no_grad():
                            fake_images = self.G(noise)
                            
                        # Real images loss
                        D_real = self.D(real).view(-1)
                        # Use soft real labels
                        real_label = torch.full_like(D_real, 0.9)
                        if self.config.use_distortion:
                            real_label = real_label - distortion * torch.rand_like(D_real) * 0.2
                        loss_D_real = self.criterion(D_real, real_label)
                        
                        # Fake images loss
                        D_fake = self.D(fake_images.detach()).view(-1)
                        # Use soft fake labels
                        fake_label = torch.full_like(D_fake, 0.1)
                        if self.config.use_distortion:
                            fake_label = fake_label + distortion * torch.rand_like(D_fake) * 0.2
                        loss_D_fake = self.criterion(D_fake, fake_label)
                        
                        # Combined loss
                        loss_D = loss_D_real + loss_D_fake
                        loss_D.backward()
                        self.opt_D.step()
                    else:
                        loss_D = torch.tensor(0.0, device=self.device)
                        D_paused_batches += 1
                        
                    # TRAIN GENERATOR
                    self.opt_G.zero_grad()
                    
                    # Generate fake images for generator update
                    fake_for_G = self.G(noise)
                    
                    # Get discriminator output
                    D_output = self.D(fake_for_G).view(-1)
                    
                    # Generator wants discriminator to classify fake as real
                    target_real = torch.full_like(D_output, 0.9)
                    loss_G = self.criterion(D_output, target_real)
                    
                    # Feature matching loss
                    if fake_for_G.size() == real.size():
                        feature_match_loss = torch.mean(torch.abs(
                            torch.mean(fake_for_G, dim=0) - torch.mean(real, dim=0)
                        ))
                        loss_G = loss_G + 0.1 * feature_match_loss
                        
                    # Neighbor penalty (diversity loss)
                    neighbor_penalty_lambda = self.config.neighbor_penalty_lambda if self.config.use_neighbor_penalty else 0.0
                    neighbor_penalty = 0.0
                    
                    if self.config.use_neighbor_penalty and batch_size > 1:
                        # Calculate pairwise similarity
                        fake_flat = fake_for_G.view(batch_size, -1)
                        similarity_matrix = torch.mm(fake_flat, fake_flat.t())
                        
                        # Normalize
                        norm = torch.mm(
                            torch.norm(fake_flat, dim=1).unsqueeze(1),
                            torch.norm(fake_flat, dim=1).unsqueeze(0)
                        )
                        similarity_matrix = similarity_matrix / (norm + 1e-8)
                        
                        # Create mask for off-diagonal elements
                        mask = 1.0 - torch.eye(batch_size, device=self.device)
                        
                        # Calculate penalty
                        neighbor_penalty = torch.sum(mask * torch.abs(similarity_matrix)) / (batch_size * (batch_size - 1))
                        
                    # Total generator loss
                    loss_G_total = loss_G + neighbor_penalty_lambda * neighbor_penalty
                    
                    # Update generator
                    loss_G_total.backward()
                    self.opt_G.step()
                    
                    # Record losses
                    G_losses.append(loss_G.item())
                    D_losses.append(loss_D.item())
                    
                    # Track batch time
                    batch_end = time.time()
                    batch_times.append(batch_end - data_loaded)
                    batch_start = time.time()
                    
                # Compute average losses for this epoch
                valid_D_losses = [x for x in D_losses if not (isinstance(x, float) and math.isnan(x))]
                valid_G_losses = [x for x in G_losses if not (isinstance(x, float) and math.isnan(x))]
                
                avg_D_loss = sum(valid_D_losses) / max(1, len(valid_D_losses)) if valid_D_losses else float('nan')
                avg_G_loss = sum(valid_G_losses) / max(1, len(valid_G_losses)) if valid_G_losses else float('nan')
                
                # End of epoch timing
                epoch_end = time.time()
                
                # Logging
                print(f"[E{epoch+1}] D: {avg_D_loss:.4f} | G: {avg_G_loss:.4f} | D-paused: {D_paused_batches}/{len(train_loader)} | Time: {epoch_end-epoch_start:.1f}s")
                
                # Performance logging
                avg_data_time = sum(data_times)/len(data_times) if data_times else 'N/A'
                avg_batch_time = sum(batch_times)/len(batch_times) if batch_times else 'N/A'
                print(f"[TIMING] Epoch {epoch+1}: Total {epoch_end-epoch_start:.2f}s | Avg data load {avg_data_time}s | Avg train {avg_batch_time}s per batch")
                print(f"[LR] G: {self.opt_G.param_groups[0]['lr']:.6f} | D: {self.opt_D.param_groups[0]['lr']:.6f}")
                
                # Save sample and checkpoint at intervals
                if (epoch + 1) % self.config.sample_interval == 0 or epoch == 0:
                    print(f"[Epoch {epoch+1}] Saving sample and checkpoint...")
                    self.generate_sample(epoch)
                    self.save_checkpoint(epoch)
                    
                    # Also save the resolution-specific checkpoint
                    self.save_checkpoint(epoch, resolution=self.img_size)
                    
                # Save full model at intervals
                if (epoch + 1) % self.config.save_epoch_checkpoint_interval == 0:
                    self.save_full_model(epoch)
                    
                # Check for pixel-perfect match
                if self.config.pixel_perfect_matching and self.reference_image_tensor is not None:
                    self.G.eval()
                    with torch.no_grad():
                        gen_img = self.G(self.fixed_seed_noise)
                    real_img = self.reference_image_tensor.to(self.device)
                    if self._images_equal(gen_img, real_img):
                        print(f"[MATCH] Generator output matches reference image at epoch {epoch+1}!")
                        self._save_pixelmatch_outputs(epoch, gen_img, real_img)
                        print("[MATCH] Stopping training early.")
                        return True  # Stop training
                        
                # Increment epoch counter
                epoch += 1
                vram_retry = 0  # Reset VRAM retry counter
                
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"[VRAM][OOM] CUDA out of memory detected at epoch {epoch+1}. Adjusting memory settings...")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    # Increase checkpointing level
                    self.checkpointing_level += 1
                    self.enable_gradient_checkpointing()
                    print(f"[VRAM] Aggressiveness level: {self.checkpointing_level}")
                    
                    # Reduce batch size further if possible
                    if self.auto_reduce_batch_size and hasattr(train_loader, 'batch_sampler') and train_loader.batch_sampler is not None:
                        current_batch_size = train_loader.batch_sampler.batch_size
                        if current_batch_size > 1:
                            new_batch_size = current_batch_size // 2
                            print(f"[VRAM] Reducing batch size from {current_batch_size} to {new_batch_size}")
                            # Need to recreate data loader
                            train_loader = self.create_dataloader(self.img_size)
                    
                    vram_retry += 1
                    if vram_retry > self.max_vram_retries:
                        print("[VRAM] Too many VRAM fallback attempts. Exiting training.")
                        raise
                    continue  # Retry epoch
                else:
                    raise  # Re-raise other exceptions
                    
        print(f"Training complete. Reached epoch {epoch} at resolution {self.img_size}x{self.img_size}")
        return True
        
    def train_progressive(self, start_resolution_idx=0, max_resolution_idx=None, epochs_per_resolution=None):
        """Train the GAN with progressive growing through multiple resolutions"""
        # Validate resolution indices
        if start_resolution_idx < 0 or start_resolution_idx >= len(self.config.resolutions):
            start_resolution_idx = 0
            
        if max_resolution_idx is None:
            max_resolution_idx = len(self.config.resolutions) - 1
        elif max_resolution_idx < start_resolution_idx or max_resolution_idx >= len(self.config.resolutions):
            max_resolution_idx = len(self.config.resolutions) - 1
            
        # Train through each resolution
        for res_idx in range(start_resolution_idx, max_resolution_idx + 1):
            resolution = self.config.resolutions[res_idx]
            print(f"\n{'='*50}")
            print(f"STARTING TRAINING AT RESOLUTION {resolution}x{resolution}")
            print(f"{'='*50}\n")
            
            # Try to load a checkpoint for this resolution
            checkpoint_loaded = self.load_checkpoint(resolution=resolution)
            
            # If no checkpoint for this resolution but we're not at the lowest resolution,
            # try to load the previous resolution and grow
            if not checkpoint_loaded and res_idx > 0:
                prev_resolution = self.config.resolutions[res_idx - 1]
                prev_checkpoint_loaded = self.load_checkpoint(resolution=prev_resolution)
                
                if prev_checkpoint_loaded:
                    print(f"[PROGRESSIVE] Loaded checkpoint from lower resolution {prev_resolution}x{prev_resolution}")
                    # Grow the model to current resolution
                    self.grow_to_resolution(resolution)
                    
            # Get epochs for this resolution
            if epochs_per_resolution is not None and resolution in epochs_per_resolution:
                epochs = epochs_per_resolution[resolution]
            else:
                epochs = self.config.min_epochs_per_res.get(resolution, 100)
                
            # Train at this resolution
            self.train(epochs=epochs, resolution=resolution)
            
            # Save final checkpoint for this resolution
            self.save_checkpoint(self.start_epoch, resolution=resolution)
            self.save_full_model(self.start_epoch)
            
        print("\nProgressive training complete!")
        return True


def main():
    """Main function to train the GAN using settings from config.yaml"""
    import argparse
    
    # Parse command line arguments (optional overrides for config file)
    parser = argparse.ArgumentParser(description='Train a progressive GAN or generate samples')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--command', type=str, 
                        choices=['train', 'sample', 'progressive', 'save_full_model'], 
                        help='Override command from config file')
    parser.add_argument('--resolution', type=int, help='Override resolution from config file')
    parser.add_argument('--epochs', type=int, help='Override epochs from config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = GANConfig(args.config)
    
    # Command-line args override config file settings
    if args.command:
        config.training_command = args.command
    if args.resolution:
        config.resolution = args.resolution
    if args.epochs:
        config.epochs = args.epochs
      # Print system info
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU")
        
    print(f"[CONFIG] Loaded from: {args.config}")
    print(f"[CONFIG] Training command: {config.training_command}")
    
    # Create trainer
    trainer = GANTrainer(config)
    
    # Process command from config
    if config.training_command == 'sample':
        # Just generate a sample from an existing model
        resolution = config.resolution if hasattr(config, 'resolution') and config.resolution else config.resolutions[-1]
        print(f"Generating samples at resolution {resolution}x{resolution}...")
        
        # Load existing model
        trainer.load_checkpoint(resolution=resolution)
        trainer.generate_sample(epoch=0)
        
    elif config.training_command == 'train':
        # Train at a fixed resolution
        resolution = config.resolution if hasattr(config, 'resolution') and config.resolution else config.resolutions[0]
        epochs = config.epochs if hasattr(config, 'epochs') and config.epochs else config.min_epochs_per_res.get(resolution, 100)
        
        print(f"Training at resolution {resolution}x{resolution} for {epochs} epochs...")
        trainer.train(epochs=epochs, resolution=resolution)
        
    elif config.training_command == 'progressive':
        # Progressive growing through multiple resolutions
        start_idx = config.start_res_idx
        max_idx = config.max_res_idx if config.max_res_idx is not None else len(config.resolutions) - 1
        
        # Print training plan
        print("Progressive training plan:")
        for i in range(start_idx, max_idx + 1):
            res = config.resolutions[i]
            epochs = config.min_epochs_per_res.get(res, 100)
            print(f"  - Resolution {res}x{res}: {epochs} epochs")
            
        # Execute progressive training
        trainer.train_progressive(start_resolution_idx=start_idx, max_resolution_idx=max_idx)
        
    elif config.training_command == 'save_full_model':
        # Load model and save a full model file
        resolution = config.resolution if hasattr(config, 'resolution') and config.resolution else config.resolutions[-1]
        print(f"Loading model at resolution {resolution}x{resolution} and saving full model...")
        
        # Load existing model
        trainer.load_checkpoint(resolution=resolution)
        trainer.save_full_model(epoch=0, manual=True)
        
    print("Done!")

if __name__ == "__main__":
    # Enable command-line execution
    main()
