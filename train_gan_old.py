import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image
from models.gan_modules import Generator, Discriminator
from models.git_model_handler import GitModelHandler
from tqdm import tqdm
import argparse
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from models.gan_weight_transfer import transfer_gan_weights, get_best_practice_scheduler
from torchvision.utils import save_image

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- CONFIG ---
# Progressive resolution settings
RESOLUTIONS = [32, 64, 128, 256, 512, 720, 1080]  # Now starts at 32
START_RES_INDEX = 0  # Start at lowest resolution
MAX_RES_INDEX = len(RESOLUTIONS) - 1
# Minimum epochs per resolution for a dataset of 1000 images with augmentation
MIN_EPOCHS_PER_RES = {
    32: 100,
    64: 200,
    128: 300,
    256: 400,
    512: 600,
    720: 800,
    1080: 1200
}
# Use absolute path to avoid path resolution issues
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMG_SIZE = 720  # Always train at 720p
BATCH_SIZE = 4
EPOCHS = 50000  # Best practice: use clear, descriptive variable name for epochs
LEARNING_RATE = 2e-4
Z_DIM = 100
CHECKPOINT_PATH = os.path.join(DATA_DIR, 'gan_checkpoint.pth')
SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
FULL_MODEL_PATH = os.path.join(DATA_DIR, 'gan_full_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_INTERVAL = 100

os.makedirs(SAMPLE_DIR, exist_ok=True)

# For handling memory constraints based on resolution
def get_adjusted_batch_size(resolution):
    if resolution >= 720:
        return 4  # Very small batch size for 720p and higher
    elif resolution >= 512:
        return 8  # Small batch size for 512p
    elif resolution >= 256:
        return 16  # Medium batch size for 256p
    else:
        return 32  # Default batch size for low resolutions

# --- DATASET ---
class AxolotlDataset(Dataset):
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

# --- DATASET TRANSFORM FACTORY ---
def get_transform(img_size, augment=True):
    if not augment:
        # No augmentation: just resize, center crop, tensor, normalize
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    # Data augmentation: H flip, small rotation, brightness/contrast, center crop/pad
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # Only left-right flip
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.CenterCrop(img_size),  # Ensures center crop
        transforms.Pad(padding=4, fill=0, padding_mode='constant'),  # Small pad, then crop again
        transforms.CenterCrop(img_size),  # Crop back to original size
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

class AugmentedDataset(IterableDataset):
    """
    IterableDataset that yields 20,000 unique, randomly augmented samples per epoch.
    Each sample is a random image from the dataset with random augmentation applied.
    """
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

transform = get_transform(IMG_SIZE)
train_ds = AugmentedDataset(TRAIN_DIR, transform, target_size=20000)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# --- GAN TRAINER CLASS ---
import hashlib

class GANTrainer:
    def __init__(self, img_size=128, z_dim=100, lr=2e-4, batch_size=32, device='cpu', use_neighbor_penalty=True, use_distortion=True, reference_image_path=None, fixed_seed=1234):
        self.img_size = img_size
        self.z_dim = z_dim
        self.device = device
        self.use_neighbor_penalty = use_neighbor_penalty
        self.use_distortion = use_distortion
        self.G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
        self.D = Discriminator(img_channels=3, img_size=img_size).to(device)
        # Set learning rates: G=0.0002, D=0.00005 (D much lower)
        self.opt_G = optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=0.00001, betas=(0.5, 0.999))
        # Store initial learning rates for resuming
        for param_group in self.opt_G.param_groups:
            param_group['lr_init'] = 0.0002
        for param_group in self.opt_D.param_groups:
            param_group['lr_init'] = 0.0001
        # Remove learning rate schedulers for G and D (no LR decay)
        self.scheduler_G = None
        self.scheduler_D = None
        print(f"[INFO] Self-adjusting learning rate is ENABLED for both Generator and Discriminator.")
        self.criterion = nn.BCELoss()
        self.start_epoch = 0
        self.fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
        self.overfit_counter = 0
        self.overfit_patience = 100
        self.last_D_losses = []
        self.last_G_losses = []
        self.checkpointing_level = 0
        self.git_enabled = False
        self.git_push_interval = 1000
        self.sample_interval = 100
        self.resolution_history = []
        self.res_index = RESOLUTIONS.index(img_size) if img_size in RESOLUTIONS else 0
        self.epochs_at_res = 0  # Track epochs at current resolution
        print(f"[SUMMARY] GANTrainer will start at {img_size}x{img_size} and train only at this resolution.")
        self.reference_image_path = reference_image_path
        self.fixed_seed = fixed_seed
        self.reference_image_tensor = None
        if reference_image_path is not None:
            self.reference_image_tensor = self._load_reference_image(reference_image_path)
        self.fixed_seed_noise = self._get_fixed_seed_noise()

    def _load_reference_image(self, path):
        # Load and preprocess the reference image to tensor (normalized like training images)
        img = Image.open(path).convert('RGB')
        transform = get_transform(self.img_size, augment=False)
        return transform(img).unsqueeze(0)  # Add batch dim

    def _get_fixed_seed_noise(self):
        # Always use the same fixed seed for reproducibility
        g = torch.Generator(device=self.device)
        g.manual_seed(self.fixed_seed)
        return torch.randn(1, self.z_dim, 1, 1, device=self.device, generator=g)

    def _images_equal(self, img1, img2):
        # img1, img2: torch tensors, shape (1, 3, H, W), normalized [-1,1]
        # Convert to uint8 for strict pixel comparison
        def to_uint8(t):
            t = t.detach().cpu()
            t = (t * 0.5 + 0.5).clamp(0, 1) * 255
            return t.round().to(torch.uint8)
        return torch.equal(to_uint8(img1), to_uint8(img2))

    def _save_pixelmatch_outputs(self, epoch, gen_img, real_img):
        # Save the fixed seed, model, and both images
        out_dir = os.path.join(SAMPLE_DIR, 'pixelmatch_success')
        os.makedirs(out_dir, exist_ok=True)
        # Save seed
        with open(os.path.join(out_dir, 'fixed_seed.txt'), 'w') as f:
            f.write(str(self.fixed_seed))
        # Save model
        self.save_full_model(epoch, manual=True)
        # Save images
        save_image(gen_img, os.path.join(out_dir, 'generated.png'), normalize=True)
        save_image(real_img, os.path.join(out_dir, 'reference.png'), normalize=True)
        print(f"[SUCCESS] Pixel-perfect match found at epoch {epoch+1}. Outputs saved to {out_dir}")

    def _check_git_available(self):
        """Check if Git is available and we're in a Git repository"""
        try:
            # Try to execute a simple git command
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"], 
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM memory at the cost of computation speed"""
        try:
            # Import improved gradient checkpointing module
            from models.checkpoint_utils import enable_gradient_checkpointing
            
            # Enable checkpointing on Generator with current level
            enable_gradient_checkpointing(self.G, level=self.checkpointing_level)
            print(f"[VRAM] Enabled gradient checkpointing on Generator (level {self.checkpointing_level})")
            
            # Enable checkpointing on Discriminator
            enable_gradient_checkpointing(self.D, level=self.checkpointing_level)
            print(f"[VRAM] Enabled gradient checkpointing on Discriminator (level {self.checkpointing_level})")
            
            # Additional memory saving measures based on checkpoint level
            if self.checkpointing_level >= 2:
                # Reduce batch size
                global BATCH_SIZE
                new_batch_size = max(4, BATCH_SIZE // 2)
                if new_batch_size != BATCH_SIZE:
                    print(f"[VRAM] Reducing batch size from {BATCH_SIZE} to {new_batch_size}")
                    BATCH_SIZE = new_batch_size
                    
            if self.checkpointing_level >= 3:
                # Use mixed precision training if available
                try:
                    from torch.cuda.amp import autocast, GradScaler
                    self.use_amp = True
                    self.scaler = GradScaler()
                    print("[VRAM] Enabled mixed precision training")
                except ImportError:
                    print("[VRAM] Mixed precision training not available")
                    
            return True
        except Exception as e:
            print(f"[VRAM] Error enabling gradient checkpointing: {str(e)}")
            return False

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing to restore normal forward pass"""
        try:
            # Import our improved utility
            from models.checkpoint_utils import disable_gradient_checkpointing
            
            # Disable checkpointing on both models
            disable_gradient_checkpointing(self.G)
            disable_gradient_checkpointing(self.D)
            
            print("[VRAM] Disabled gradient checkpointing")
            return True
        except Exception as e:
            print(f"[VRAM] Error disabling gradient checkpointing: {str(e)}")
            return False

    def save_checkpoint(self, epoch):
        """Save model checkpoint and push to Git at specified intervals"""
        # Get clean model states first
        self.disable_gradient_checkpointing()
        
        # Use a temporary checkpoint path to avoid corrupting the main checkpoint if saving fails
        temp_checkpoint_path = CHECKPOINT_PATH + ".tmp"
        try:
            # Save the checkpoint to temporary file first
            torch.save({
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'opt_G': self.opt_G.state_dict(),
                'opt_D': self.opt_D.state_dict(),
                'epoch': epoch,
                'img_size': self.img_size  # Store current image size for proper restoration
            }, temp_checkpoint_path)
            
            # Rename temp file to actual checkpoint path (safer file operation)
            if os.path.exists(temp_checkpoint_path):
                import shutil
                shutil.move(temp_checkpoint_path, CHECKPOINT_PATH)
                
            print(f"[Checkpoint] Saved checkpoint at epoch {epoch+1}")
                
            # Push to Git every git_push_interval epochs
            # if self.git_enabled and (epoch + 1) % self.git_push_interval == 0:
            #     print(f"[GIT] Pushing model checkpoint at epoch {epoch+1} to Git main branch...")
            #     try:
            #         self.git_handler.update_model_in_git(epoch_num=epoch+1)
            #     except Exception as e:
            #         print(f"[GIT] Warning: Failed to push model to Git: {str(e)}")
            #         print("[GIT] Continuing training without Git push")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {str(e)}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)
        
        # Log resolution history
        self.resolution_history.append({'epoch': epoch+1, 'img_size': self.img_size})
        print(f"[CHECKPOINT] Epoch {epoch+1} | Resolution: {self.img_size}x{self.img_size}")
        # Save a new checkpoint file every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            checkpoint_path_epoch = os.path.join(DATA_DIR, f'gan_checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'opt_G': self.opt_G.state_dict(),
                'opt_D': self.opt_D.state_dict(),
                'epoch': epoch,
                'img_size': self.img_size
            }, checkpoint_path_epoch)
            print(f"[CHECKPOINT] Saved epoch checkpoint: {checkpoint_path_epoch}")

    def save_full_model(self, epoch, manual=False):
        """Save the full model (not just checkpoint) and push to Git"""
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Make sure we're using absolute paths
        full_model_abs_path = os.path.abspath(FULL_MODEL_PATH)
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
        # Push only the full model to git
        # if self.git_enabled:
        #     try:
        #         handler = GitModelHandler(FULL_MODEL_PATH)
        #         comment = f"Full GAN model at epoch {epoch+1}" if not manual else "Manual full GAN model save"
        #         handler.update_model_in_git(epoch_num=epoch+1)
        #         print(f"[GIT] Pushed full model to Git with comment: {comment}")
        #     except Exception as e:
        #         print(f"[GIT] Warning: Failed to push full model to Git: {str(e)}")

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            try:
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
                # Use the specified image size from initialization, not hardcoded 720
                self.G = Generator(z_dim=self.z_dim, img_channels=3, img_size=self.img_size).to(self.device)
                self.D = Discriminator(img_channels=3, img_size=self.img_size).to(self.device)
                # Set learning rates: G=0.0002, D=0.00005 (D much lower)
                self.opt_G = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
                self.opt_D = optim.Adam(self.D.parameters(), lr=0.00005, betas=(0.5, 0.999))
                self.fixed_noise = torch.randn(16, self.z_dim, 1, 1, device=self.device)
                
                # Don't try to load weights if resolution mismatch
                checkpoint_img_size = checkpoint.get('img_size', 720)
                if checkpoint_img_size == self.img_size:
                    # Same resolution, safe to load the weights
                    self.G.load_state_dict(checkpoint['G'])
                    self.D.load_state_dict(checkpoint['D'])
                    self.opt_G.load_state_dict(checkpoint['opt_G'])
                    self.opt_D.load_state_dict(checkpoint['opt_D'])
                    # Make sure we're not using gradient checkpointing when loading
                    self.disable_gradient_checkpointing()
                    self.start_epoch = checkpoint['epoch'] + 1
                    print(f"[INFO] Successfully loaded checkpoint from {CHECKPOINT_PATH} at epoch {self.start_epoch}")
                else:
                    # Resolution mismatch, starting from scratch
                    print(f"[WARN] Resolution mismatch: Checkpoint is for {checkpoint_img_size}x{checkpoint_img_size}, but training at {self.img_size}x{self.img_size}. Starting from scratch.")
                    self.start_epoch = 0
            except Exception as e:
                print(f"[WARN] Error loading checkpoint: {str(e)}. Starting from scratch.")
                self.start_epoch = 0
        else:
            print("No checkpoint found, starting fresh.")
            self.start_epoch = 0
            # If checkpoint does not exist, create new models from scratch
            if not os.path.exists(CHECKPOINT_PATH):
                print("[INFO] No checkpoint found, initializing new models from scratch.")
                self.G = Generator(z_dim=Z_DIM, img_channels=3, img_size=self.img_size).to(self.device)
                self.D = Discriminator(img_channels=3, img_size=self.img_size).to(self.device)
                self.opt_G = optim.Adam(self.G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
                self.opt_D = optim.Adam(self.D.parameters(), lr=LEARNING_RATE * 0.2, betas=(0.5, 0.999))  # D learns much slower than G
                self.fixed_noise = torch.randn(16, self.z_dim, 1, 1, device=self.device)
                self.start_epoch = 0
                return

    def generate_sample(self, epoch):
        """Generate a sample using VRAM-safe approach"""
        import torch.nn.functional as F
        try:
            # Set eval mode and disable checkpointing to ensure clean output
            self.G.eval()
            # Use our new method to disable gradient checkpointing

            self.disable_gradient_checkpointing()

            # Generate single image first (smaller memory footprint)
            with torch.no_grad():
                fake = self.G(self.fixed_noise[:1]).detach().cpu()
                save_image(fake, os.path.join(SAMPLE_DIR, f'sample_epoch.png'), normalize=True)

                # Generate a grid of samples and overwrite the same file
                try:
                    torch.cuda.empty_cache()  # Clear memory first
                    # Use truly new random noise for each image in the grid
                    grid_noise = torch.randn(16, self.z_dim, 1, 1, device=self.device)
                    fake_grid = self.G(grid_noise).detach().cpu()
                    # Each image in the grid is 720x720, no upscaling or resizing
                    save_image(fake_grid, os.path.join(SAMPLE_DIR, 'sample_grid.png'), normalize=True, nrow=4, padding=2, pad_value=1)
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print("[VRAM] Grid sample generation skipped due to memory constraints")
                    else:
                        print(f"[ERROR] Grid sample generation error: {str(e)}")
        except Exception as e:
            print(f"[ERROR] Sample generation failed: {str(e)}")
            # Fall back to CPU if necessary
            try:
                print("[VRAM] Attempting to generate sample on CPU...")
                self.G = self.G.cpu()
                with torch.no_grad():
                    fake = self.G(self.fixed_noise[:1].cpu()).detach()
                    save_image(fake, os.path.join(SAMPLE_DIR, 'sample_grid.png'), normalize=True)
                self.G = self.G.to(self.device)
            except Exception as e2:
                print(f"[CRITICAL] CPU fallback failed too: {str(e2)}")
        finally:
            # Restore model to training state
            self.G.train()
            # Re-enable gradient checkpointing if it was active
            if hasattr(self, 'checkpointing_level') and self.checkpointing_level > 0:
                self.enable_gradient_checkpointing()

    def grow_to_resolution(self, new_img_size):
        """Grow the current model to a new resolution, copying over weights in-place."""
        if new_img_size == self.img_size:
            print(f"[GROW] Model already at {new_img_size}x{new_img_size}.")
            return
        print(f"[GROW] Growing model from {self.img_size}x{self.img_size} to {new_img_size}x{new_img_size}...")
        from models.gan_modules import Generator, Discriminator
        from models.gan_weight_transfer import transfer_gan_weights
        # Create new models at the higher resolution
        new_G = Generator(z_dim=self.z_dim, img_channels=3, img_size=new_img_size).to(self.device)
        new_D = Discriminator(img_channels=3, img_size=new_img_size).to(self.device)
        # Transfer weights from old to new
        transfer_gan_weights(self.G, new_G)
        transfer_gan_weights(self.D, new_D)
        # Replace models in-place
        self.G = new_G
        self.D = new_D
        self.img_size = new_img_size
        print(f"[GROW] Model grown to {new_img_size}x{new_img_size} and weights transferred.")
        # Re-create optimizers for new parameters
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=0.00005, betas=(0.5, 0.999))
        self.fixed_seed_noise = self._get_fixed_seed_noise()

    def train(self, train_loader, epochs=float('inf'), sample_interval=1000):
        """Fixed training loop to resolve indentation issues and improve training"""
        local_train_loader = train_loader
        self.load_checkpoint()
        epoch = self.start_epoch
        best_val = float('inf')
        no_improve = 0
        prev_G_loss = None
        vram_retry = 0
        max_epochs = epochs
        self.epochs_at_res = 0  # Track epochs at current resolution
        manual_distortion_scale = 0.2 # Set your fixed distortion value here
        distortion = manual_distortion_scale if self.use_distortion else 0.0  # Use full value, no cap
        avg_D_loss = None  # Initialize to None to avoid UnboundLocalError
        avg_G_loss = None  # Initialize to None to avoid UnboundLocalError

        # --- VRAM/BATCH SIZE AUTO-ADJUST FOR HIGH RES ---
        if self.img_size >= 720:
            print(f"[VRAM] High resolution detected ({self.img_size}x{self.img_size}). Auto-reducing batch size and learning rates.")
            # Reduce batch size to minimum (1)
            for param_group in self.opt_G.param_groups:
                param_group['lr'] = 0.00005
            for param_group in self.opt_D.param_groups:
                param_group['lr'] = 0.00001
            # If DataLoader supports, reduce batch size (user must restart for full effect)
            if hasattr(train_loader, 'batch_size'):
                print(f"[VRAM] (Note: DataLoader batch_size is {train_loader.batch_size}. For best results, set batch_size=1 in DataLoader at 720p+)")

        while epoch < max_epochs:
            # Set a new random seed for each epoch for better randomness
            epoch_seed = int(time.time()) + epoch
            random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            np.random.seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(epoch_seed)
            
            try:
                # Enable anomaly detection temporarily to identify the exact operation causing the issue
                torch.autograd.set_detect_anomaly(True)
                D_losses = []
                G_losses = []
                D_paused_batches = 0
                G_paused_batches = 0
                D_paused_streak = 0
                G_paused_streak = 0
                max_D_paused_streak = 0
                max_G_paused_streak = 0
                last_paused = None
                epoch_start = time.time()
                pbar = tqdm(local_train_loader, desc=f"Epoch {epoch+1} [{self.img_size}x{self.img_size}]", leave=False)
                use_amp = hasattr(self, 'use_amp') and self.use_amp
                batch_times = []
                data_times = []
                batch_start = time.time()
                distortion_values = []  # Track distortion values for this epoch (for logging only)
                example_distortion_saved = False
                
                # Process each batch in the epoch
                for real in pbar:
                    data_loaded = time.time()
                    data_times.append(data_loaded - batch_start)
                    real = real.to(self.device)
                    batch_size = real.size(0)
                    noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                    
                    # Set fixed learning rates
                    for param_group in self.opt_G.param_groups:
                        param_group['lr'] = 0.0002  # Slightly higher LR for generator
                    for param_group in self.opt_D.param_groups:
                        param_group['lr'] = 0.00005  # Lower LR for discriminator to slow it down
                    
                    # 1. Train Discriminator: train less frequently than generator
                    # Only update discriminator every other batch to prevent it from overpowering the generator
                    train_discriminator = True
                    # Only use avg_D_loss/avg_G_loss if they are not None
                    if epoch > 0 and avg_D_loss is not None and avg_G_loss is not None:
                        if avg_D_loss < 0.3 and avg_D_loss < avg_G_loss / 2:
                            # If discriminator is too strong, skip training it this batch
                            train_discriminator = random.random() > 0.5
                    
                    if train_discriminator:
                        self.opt_D.zero_grad()
                        
                        # Generate fake images
                        with torch.no_grad():  # Don't track gradients here to save memory
                            fake_images = self.G(noise)
                        
                        # Real images loss - add noise to labels for smoother training
                        D_real = self.D(real).view(-1)
                        # Use soft real labels between 0.7 and 1.0 rather than exactly 1
                        real_label = torch.full_like(D_real, 0.9)
                        if self.use_distortion:
                            real_label = real_label - distortion * torch.rand_like(D_real) * 0.2  # smaller noise
                        loss_D_real = self.criterion(D_real, real_label)
                        
                        # Fake images loss - add noise to labels for smoother training
                        D_fake = self.D(fake_images.detach()).view(-1)
                        # Use soft fake labels between 0.0 and 0.3 rather than exactly 0
                        fake_label = torch.full_like(D_fake, 0.1)
                        if self.use_distortion:
                            fake_label = fake_label + distortion * torch.rand_like(D_fake) * 0.2  # smaller noise
                        loss_D_fake = self.criterion(D_fake, fake_label)
                        
                        # Combined discriminator loss
                        loss_D = loss_D_real + loss_D_fake
                        loss_D.backward()
                        self.opt_D.step()
                    else:
                        loss_D = torch.tensor(0.0, device=self.device)
                        D_paused_batches += 1
                    
                    # 2. Train Generator: always train the generator
                    self.opt_G.zero_grad()
                    
                    # Generate new fake images for generator update (don't reuse)
                    fake_for_G = self.G(noise)
                    
                    # Feed fake images into discriminator
                    D_output = self.D(fake_for_G).view(-1)
                    
                    # Generator wants discriminator to classify fake images as real
                    # Target for generator is always 1 (real)
                    target_real = torch.full_like(D_output, 0.9)  # Use 0.9 instead of 1.0 for smoother training
                    loss_G = self.criterion(D_output, target_real)
                    
                    # Feature matching loss - helps generator match statistics of real data
                    if isinstance(fake_for_G, torch.Tensor) and isinstance(real, torch.Tensor) and fake_for_G.size() == real.size():
                        # Calculate feature matching loss: make fake images statistically similar to real ones
                        feature_match_loss = torch.mean(torch.abs(
                            torch.mean(fake_for_G, dim=0) - torch.mean(real, dim=0)
                        ))
                        loss_G = loss_G + 0.1 * feature_match_loss  # Add small weight to feature matching
                        
                        # Use fixed distortion for all batches
                        # Only save example-distortion.png once per epoch
                        if not example_distortion_saved:
                            from torchvision.utils import save_image
                            real_img = real[0].detach().cpu()
                            if self.use_distortion and distortion > 0.0:
                                # Add random noise proportional to distortion
                                noise = torch.randn_like(real_img) * distortion
                                distorted_img = real_img + noise
                                distorted_img = torch.clamp(distorted_img, -1.0, 1.0)
                                save_image(distorted_img, os.path.join(SAMPLE_DIR, 'example-distortion.png'), normalize=True)
                                print(f"[DISTORTION] Saved example-distortion.png with distortion={distortion:.4f} (noise)")
                            else:
                                save_image(real_img, os.path.join(SAMPLE_DIR, 'example-distortion.png'), normalize=True)
                                print(f"[DISTORTION] Saved example-distortion.png with distortion=0 (no distortion)")
                            example_distortion_saved = True
                        
                        # Remove per-batch distortion log; accumulate for epoch
                        if 'distortion_accum' not in locals():
                            distortion_accum = 0.0
                            distortion_batches = 0
                        distortion_accum += distortion
                        distortion_batches += 1
                        
                    # Neighbor penalty: encourage diverse outputs
                    neighbor_penalty_lambda = 0.5 if self.use_neighbor_penalty else 0.0
                    neighbor_penalty = 0.0
                    
                    if self.use_neighbor_penalty and batch_size > 1:
                        # Calculate pairwise differences between generated images
                        # This encourages the generator to produce diverse images
                        fake_flat = fake_for_G.view(batch_size, -1)
                        similarity_matrix = torch.mm(fake_flat, fake_flat.t())
                        
                        # Normalize the similarity matrix
                        norm = torch.mm(
                            torch.norm(fake_flat, dim=1).unsqueeze(1),
                            torch.norm(fake_flat, dim=1).unsqueeze(0)
                        )
                        similarity_matrix = similarity_matrix / (norm + 1e-8)
                        
                        # We want off-diagonal elements to be small (dissimilar images)
                        # Create a mask for the off-diagonal elements
                        mask = 1.0 - torch.eye(batch_size, device=self.device)
                        
                        # Calculate penalty based on similarity
                        neighbor_penalty = torch.sum(mask * torch.abs(similarity_matrix)) / (batch_size * (batch_size - 1))
                        
                        # Apply the penalty to encourage diversity
                        similarity_stats = {
                            "mean": similarity_matrix.mean().item(),
                            "max": similarity_matrix.max().item(),
                            "min": similarity_matrix.min().item()
                        }
                        
                    # Apply both losses to generator
                    loss_G_total = loss_G + neighbor_penalty_lambda * neighbor_penalty
                    
                    # Actually apply the Generator loss gradient and update weights
                    loss_G_total.backward()
                    self.opt_G.step()
                    
                    # Record losses for logging
                    G_losses.append(loss_G.item())
                    D_losses.append(loss_D.item())
                    
                    batch_end = time.time()
                    batch_times.append(batch_end - data_loaded)
                    batch_start = time.time()
                
                # End of batch for-loop
                
                # Compute average losses, ignoring NaNs
                # Disable anomaly detection to restore performance
                torch.autograd.set_detect_anomaly(False)
                avg_D_loss = (sum([x for x in D_losses if not (isinstance(x, float) and x != x)]) / max(1, len([x for x in D_losses if not (isinstance(x, float) and x != x)]))) if any([not (isinstance(x, float) and x != x) for x in D_losses]) else float('nan')
                avg_G_loss = (sum([x for x in G_losses if not (isinstance(x, float) and x != x)]) / max(1, len([x for x in G_losses if not (isinstance(x, float) and x != x)]))) if any([not (isinstance(x, float) and x != x) for x in G_losses]) else float('nan')
                
                # Log average distortion for this epoch
                if distortion_values:
                    avg_distortion = sum(distortion_values) / len(distortion_values)
                    print(f"[DISTORTION][E{epoch+1}] Avg distortion: {avg_distortion:.4f} (manual_scale={manual_distortion_scale})")
                
                epoch_end = time.time()
                
                # Only one of D or G is paused per batch; sum equals number of batches
                num_batches = D_paused_batches + G_paused_batches
                print(f"[E{epoch+1}] D: {avg_D_loss:.4f} | G: {avg_G_loss:.4f} | D-paused: {D_paused_batches}/{num_batches} (streak {D_paused_streak}) | G-paused: {G_paused_batches}/{num_batches} (streak {G_paused_streak}) | Time: {epoch_end-epoch_start:.1f}s")
                
                # --- Fix: Avoid ZeroDivisionError if no batches processed ---
                avg_data_time = sum(data_times)/len(data_times) if data_times else 'N/A'
                avg_batch_time = sum(batch_times)/len(batch_times) if batch_times else 'N/A'
                print(f"[TIMING] Epoch {epoch+1}: Total {epoch_end-epoch_start:.2f}s | Avg data load {avg_data_time}s | Avg train {avg_batch_time}s per batch")
                print(f"[LR] G: {self.opt_G.param_groups[0]['lr']:.6f} | D: {self.opt_D.param_groups[0]['lr']:.6f}")
                
                # Save sample and checkpoint at intervals
                if (epoch + 1) % self.sample_interval == 0 or epoch == 0:
                    print(f"[Epoch {epoch+1}] Saving sample and checkpoint...")
                    self.generate_sample(epoch+1)
                    self.save_checkpoint(epoch)
                    # Also update the manual sample image (now on the same schedule as checkpoints)
                    self.G.eval()
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise[:1]).detach().cpu()
                        from torchvision.utils import save_image
                        save_image(fake, os.path.join(SAMPLE_DIR, 'sample_epochmanual.png'), normalize=True)
                    self.G.train()
                    
                    # --- Save a grid of real images as seen by the Discriminator ---
                    try:
                        real_batch = next(iter(train_loader))
                        real_grid_path = os.path.join(SAMPLE_DIR, 'real_grid_epoch.png')
                        from torchvision.utils import save_image
                        save_image(real_batch[:16], real_grid_path, normalize=True, nrow=4, padding=2, pad_value=1)
                        # Log the average distortion used this epoch in the real grid log
                        if distortion_values:
                            avg_dist = sum(distortion_values) / len(distortion_values)
                            print(f"[SAMPLE] Saved real image grid: {real_grid_path} (avg distortion: {avg_dist:.4f})")
                        else:
                            print(f"[SAMPLE] Saved real image grid: {real_grid_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to save real image grid: {e}")
                
                # Create full model file every git_push_interval epochs only
                if (epoch + 1) % self.git_push_interval == 0:
                    self.save_full_model(epoch)
                
                # Save grid image with epoch in name every 1000 epochs
                if (epoch + 1) % 1000 == 0:
                    try:
                        self.G.eval()
                        with torch.no_grad():
                            fake_grid = self.G(self.fixed_noise).detach().cpu()
                            grid_path = os.path.join(SAMPLE_DIR, f'sample_grid_epoch{epoch+1}.png')
                            from torchvision.utils import save_image
                            save_image(fake_grid, grid_path, normalize=True, nrow=4, padding=2, pad_value=1)
                            print(f"[SAMPLE] Saved grid image: {grid_path}")
                        self.G.train()
                    except Exception as e:
                        print(f"[ERROR] Failed to save grid image for epoch {epoch+1}: {e}")
                    
                    # Save full generator model every 1000 epochs with t_epoch{epoch+1}.pth
                    full_model_path = os.path.join(DATA_DIR, f'gan_full_model_t_epoch{epoch+1}.pth')
                    torch.save({
                        'G': self.G.state_dict(),
                        'opt_G': self.opt_G.state_dict(),
                        'epoch': epoch,
                        'img_size': self.img_size,
                        'resolution_history': self.resolution_history
                    }, full_model_path)
                    print(f"[FULLMODEL] Saved full generator model: {full_model_path}")
                
                # After epoch: pixel-perfect check
                if self.reference_image_tensor is not None:
                    self.G.eval()
                    with torch.no_grad():
                        gen_img = self.G(self.fixed_seed_noise)
                    real_img = self.reference_image_tensor.to(self.device)
                    if self._images_equal(gen_img, real_img):
                        print(f"[MATCH] Generator output matches reference image at epoch {epoch+1}!")
                        self._save_pixelmatch_outputs(epoch, gen_img, real_img)
                        print("[MATCH] Stopping training early.")
                        return  # Stop training
                
                epoch += 1
                vram_retry = 0  # Reset VRAM retry counter after successful epoch
            
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"[VRAM][OOM] CUDA out of memory detected at epoch {epoch+1}. Consider lowering batch size or learning rates further.")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    self.checkpointing_level += 1
                    self.enable_gradient_checkpointing()
                    print(f"[VRAM] Aggressiveness level: {self.checkpointing_level}")
                    vram_retry += 1
                    if vram_retry > 10:
                        print("[VRAM] Too many VRAM fallback attempts. Exiting training.")
                        raise
                    continue  # Retry epoch with more aggressive checkpointing
                else:
                    raise
        
        print("Training complete.")

if __name__ == '__main__':
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU")
    print(f"[PATHS] DATA_DIR: {DATA_DIR}")
    print(f"[PATHS] CHECKPOINT_PATH: {CHECKPOINT_PATH}")
    print(f"[PATHS] FULL_MODEL_PATH: {FULL_MODEL_PATH}")
    parser = argparse.ArgumentParser(description='Train GAN or generate a sample image.')
    parser.add_argument('command', nargs='?', default='train', choices=['train', 'sample', 'save_full_model'], help="'train' to train, 'sample' to generate a sample image only, 'save_full_model' to manually save and push full model")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for DataLoader (default: 4)')
    parser.add_argument('--preload', action='store_true', help='Preload all images into RAM for faster training (requires enough RAM)')
    parser.add_argument('--cache_tensors', action='store_true', help='Cache all transformed tensors in RAM for fastest data loading (no random augmentations)')
    parser.add_argument('--no-neighbor-penalty', action='store_true', help='Disable neighbor penalty in GAN training')
    parser.add_argument('--no-distortion', action='store_true', help='Disable distortion/noise in GAN training')
    parser.add_argument('--no-augment', action='store_true', help='Disable all data augmentation in training')
    parser.add_argument('--reference_image', type=str, help='Path to the reference image for pixel-perfect matching')
    parser.add_argument('--fixed_seed', type=int, default=1234, help='Fixed seed for reproducibility (default: 1234)')
    # Prompt for resolution and epochs if training
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"[INFO] CUDA: {'ON' if torch.cuda.is_available() else 'OFF'} | Device: {DEVICE}")

    if args.command == 'train':
        print(f"[INFO] Resolutions: {RESOLUTIONS}")
        
        # Load values from config.yaml
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Get resolution from config file
            res = config.get('resolution', RESOLUTIONS[0])
            if res not in RESOLUTIONS:
                print(f"[WARN] Invalid resolution in config. Using {RESOLUTIONS[0]}")
                res = RESOLUTIONS[0]
            else:
                print(f"[INFO] Using resolution {res}x{res} from config file")
                
            # Get epochs from config file
            epochs = config.get('epochs', 200)
            if not isinstance(epochs, int) or epochs < 1:
                print("[WARN] Invalid epoch count in config. Using 200.")
                epochs = 200
            else:
                print(f"[INFO] Training for {epochs} epochs from config file")
        except Exception as e:
            print(f"[ERROR] Could not load config.yaml: {e}")
            print(f"[INFO] Using default resolution {RESOLUTIONS[0]} and 200 epochs")
            res = RESOLUTIONS[0]
            epochs = 200
        
        # Adjust batch size for higher resolutions to prevent VRAM issues
        original_batch_size = BATCH_SIZE
        if res >= 512:
            adjusted_batch_size = 8
            print(f"[VRAM] Resolution {res}p detected - reducing batch size from {original_batch_size} to {adjusted_batch_size}")
            # Use a new variable instead of modifying the global
            batch_size_to_use = adjusted_batch_size
        elif res >= 256:
            adjusted_batch_size = 16
            print(f"[VRAM] Resolution {res}p detected - reducing batch size from {original_batch_size} to {adjusted_batch_size}")
            batch_size_to_use = adjusted_batch_size
        else:
            batch_size_to_use = original_batch_size
        augment = not args.no_augment
        transform = get_transform(res, augment=augment)
        train_ds = AxolotlDataset(TRAIN_DIR, transform, preload=args.preload, cache_tensors=args.cache_tensors)
        pin_memory = DEVICE.type == 'cuda'
        persistent_workers = args.num_workers > 0
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size_to_use,  # Use our adjusted batch size
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        # --- Concise progressive upscaling log and logic ---
        res_index = RESOLUTIONS.index(res)
        checkpoint_loaded = False
        # --- Reference image and seed for pixel-perfect stop ---
        reference_image_path = None
        fixed_seed = 1234
        if res == 720:
            # Use first .jpg in TRAIN_DIR as reference
            jpgs = sorted([f for f in glob.glob(os.path.join(TRAIN_DIR, '*.jpg'))])
            if jpgs:
                reference_image_path = jpgs[0]
                print(f"[MATCH] Using reference image: {reference_image_path}")
            else:
                print("[MATCH] No .jpg found in train dir; pixel-perfect stop will be disabled.")
        for lower_res in reversed(RESOLUTIONS[:res_index]):
            lower_ckpt = os.path.join(DATA_DIR, f'gan_checkpoint_{lower_res}.pth')
            if os.path.exists(lower_ckpt):
                print(f"[GROW] Growing model: {lower_res} -> {res} (loading {lower_ckpt})")
                lower_G = Generator(z_dim=Z_DIM, img_channels=3, img_size=lower_res).to(DEVICE)
                lower_D = Discriminator(img_channels=3, img_size=lower_res).to(DEVICE)
                ckpt = torch.load(lower_ckpt, map_location=DEVICE)
                lower_G.load_state_dict(ckpt['G'])
                lower_D.load_state_dict(ckpt['D'])
                trainer = GANTrainer(img_size=res, z_dim=Z_DIM, lr=LEARNING_RATE, batch_size=BATCH_SIZE, device=DEVICE, reference_image_path=reference_image_path, fixed_seed=fixed_seed)
                transfer_gan_weights(lower_G, trainer.G)
                transfer_gan_weights(lower_D, trainer.D)
                print(f"[PROGRESSIVE] Transferred weights from {lower_res} to {res}")
                checkpoint_loaded = True
                break
        if not checkpoint_loaded:
            trainer = GANTrainer(
                img_size=RESOLUTIONS[0],
                z_dim=Z_DIM,
                lr=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                use_neighbor_penalty=not args.no_neighbor_penalty,
                use_distortion=not args.no_distortion,
                reference_image_path=reference_image_path,
                fixed_seed=fixed_seed
            )
            # Progressive growing: grow in-place to the target resolution
            if res != RESOLUTIONS[0]:
                trainer.grow_to_resolution(res)
        trainer.train(train_loader, epochs=epochs, sample_interval=SAMPLE_INTERVAL)
        # --- After training, always save a checkpoint for this resolution ---
        checkpoint_path = os.path.join(DATA_DIR, f'gan_checkpoint_{res}.pth')
        torch.save({
            'G': trainer.G.state_dict(),
            'D': trainer.D.state_dict(),
            'opt_G': trainer.opt_G.state_dict(),
            'opt_D': trainer.opt_D.state_dict(),
            'epoch': getattr(trainer, 'start_epoch', 0),
            'img_size': res
        }, checkpoint_path)
        print(f"[PROGRESSIVE] Saved checkpoint for {res}x{res} at {checkpoint_path}")
    elif args.command == 'sample':
        print("Generating a full image sample using the current generator...")
        trainer = GANTrainer(img_size=RESOLUTIONS[0], z_dim=Z_DIM, lr=LEARNING_RATE, batch_size=BATCH_SIZE, device=DEVICE)
        trainer.load_checkpoint()
        try:
            trainer.generate_sample('manual')
            print(f"Sample saved to {os.path.join(SAMPLE_DIR, 'sample_epochmanual.png')}")
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("[VRAM] CUDA out of memory detected. Attempting with gradient checkpointing...")
                trainer.checkpointing_level = 1
                trainer.enable_gradient_checkpointing()
                try:
                    trainer.generate_sample('manual')
                    print(f"Sample saved to {os.path.join(SAMPLE_DIR, 'sample_epochmanual.png')}")
                except Exception as e2:
                    print(f"[ERROR] Failed to generate sample even with checkpointing: {str(e2)}")
                    print("[VRAM] Falling back to CPU generation...")
                    trainer.G = trainer.G.cpu()
                    trainer.fixed_noise = trainer.fixed_noise.cpu()
                    trainer.generate_sample('manual_cpu')
                    print(f"Sample saved to {os.path.join(SAMPLE_DIR, 'sample_epochmanual_cpu.png')}")
            else:
                print(f"[ERROR] Failed to generate sample: {e}")
    elif args.command == 'save_full_model':
        trainer = GANTrainer(img_size=RESOLUTIONS[0], z_dim=Z_DIM, lr=LEARNING_RATE, batch_size=BATCH_SIZE, device=DEVICE)
        trainer.load_checkpoint()
        trainer.save_full_model(trainer.start_epoch, manual=True)
