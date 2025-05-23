from flask import Flask, jsonify, send_file
import io
import base64
from models.generator import AxolotlGenerator
import numpy as np
from PIL import Image
from flask_cors import CORS
import os
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Ensure directories exist
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

gan = AxolotlGenerator()

class AxolotlImageAPI:
    @staticmethod
    def generate_single_image():
        # Use the exact same GAN sample command from train_gan.py for best quality
        print("=== GENERATING IMAGE USING GAN MODEL NOT DIFFUSION ===")
        try:
            import torch
            import torch.nn as nn
            from torchvision.utils import save_image
            from models.gan_modules import Generator, Discriminator
            import numpy as np
            
            # Match the exact constants from train_gan.py
            DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
            SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
            CHECKPOINT_PATH = os.path.join(DATA_DIR, 'gan_checkpoint.pth')
            IMG_SIZE = 32  # Match the IMG_SIZE in train_gan.py
            Z_DIM = 100
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create the GANTrainer instance exactly like train_gan.py
            class GANTrainer:
                def __init__(self, img_size, z_dim, device):
                    self.img_size = img_size
                    self.z_dim = z_dim
                    self.device = device
                    self.G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
                    self.fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
                    self.checkpointing_level = 0
                    
                def load_checkpoint(self):
                    if os.path.exists(CHECKPOINT_PATH):
                        try:
                            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
                            
                            # Check if the checkpoint has a different image size (exact same code as train_gan.py)
                            saved_img_size = checkpoint.get('img_size', self.img_size)
                            if saved_img_size != self.img_size:
                                print(f"[INFO] Checkpoint has image size {saved_img_size}, current is {self.img_size}")
                                print(f"[INFO] Recreating models with saved image size {saved_img_size}")
                                # Recreate models with correct size
                                self.img_size = saved_img_size
                                self.G = Generator(z_dim=self.z_dim, img_channels=3, img_size=saved_img_size).to(self.device)
                                self.fixed_noise = torch.randn(16, self.z_dim, 1, 1, device=self.device)
                            
                            # Load state dictionary
                            self.G.load_state_dict(checkpoint['G'])
                            
                            # Make sure gradient checkpointing is disabled for loaded model initially
                            if hasattr(self.G, 'gradient_checkpointing_disable'):
                                self.G.gradient_checkpointing_disable()
                                
                            print(f"[INFO] Successfully loaded checkpoint from {CHECKPOINT_PATH}")
                        except Exception as e:
                            print(f"[WARN] Error loading checkpoint: {str(e)}")
                    else:
                        print("No checkpoint found, using untrained generator")
                
                def generate_sample(self):
                    """Generate a sample image using the exact same code as train_gan.py sample command"""
                    try:
                        print("[GAN] Generating image with PyTorch GAN model using highest quality settings")
                        # Set eval mode and disable checkpointing to ensure clean output
                        self.G.eval()
                        if hasattr(self.G, 'gradient_checkpointing_disable'):
                            self.G.gradient_checkpointing_disable()
                        
                        # Generate with the exact same settings as train_gan.py sample command
                        with torch.no_grad():
                            # Directly mirror the train_gan.py sample command behavior
                            # Use fixed_noise[:1] for a single high-quality image (not a grid)
                            fake = self.G(self.fixed_noise[:1]).detach().cpu()
                            
                            # Save to memory buffer instead of disk
                            buffer = io.BytesIO()
                            # Use same normalize=True parameter as in train_gan.py
                            save_image(fake, buffer, format="PNG", normalize=True)
                            buffer.seek(0)
                            print("[GAN] Successfully generated single high-quality image")
                            return buffer.getvalue()
                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            # First try with gradient checkpointing
                            print("[VRAM] CUDA out of memory detected. Attempting with gradient checkpointing...")
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            
                            try:
                                # Enable checkpointing if available
                                self.checkpointing_level = 1
                                if hasattr(self.G, 'gradient_checkpointing_enable'):
                                    self.G.gradient_checkpointing_enable()
                                    print("[VRAM] Enabled gradient checkpointing on Generator")
                                
                                with torch.no_grad():
                                    fake = self.G(self.fixed_noise[:1]).detach().cpu()
                                    buffer = io.BytesIO()
                                    save_image(fake, buffer, format="PNG", normalize=True)
                                    buffer.seek(0)
                                    print("[GAN] Successfully generated image with gradient checkpointing")
                                    return buffer.getvalue()
                            except Exception:
                                # Last resort - move to CPU
                                print("[VRAM] Falling back to CPU generation...")
                                torch.cuda.empty_cache()
                                gc.collect()
                                
                                # Move model to CPU and generate - exact same as train_gan.py CPU fallback
                                self.G = self.G.cpu()
                                self.fixed_noise = self.fixed_noise.cpu()
                                with torch.no_grad():
                                    fake = self.G(self.fixed_noise[:1]).detach()
                                    buffer = io.BytesIO()
                                    save_image(fake, buffer, format="PNG", normalize=True)
                                    buffer.seek(0)
                                    print("[GAN] Successfully generated image on CPU")
                                    return buffer.getvalue()
                        else:
                            # Re-raise other errors
                            print(f"[ERROR] Non-memory related error: {str(e)}")
                            raise
                
                def enable_gradient_checkpointing(self):
                    """Enable gradient checkpointing to save VRAM memory at the cost of computation speed"""
                    try:
                        # Import needed for gradient checkpointing
                        import torch.utils.checkpoint
                        
                        # Enable checkpointing on Generator
                        self.G.gradient_checkpointing_enable()
                        print(f"[VRAM] Enabled gradient checkpointing on Generator (level {self.checkpointing_level})")
                            
                        return True
                    except Exception as e:
                        print(f"[VRAM] Error enabling gradient checkpointing: {str(e)}")
                        return False
            
            # Create trainer and generate sample
            trainer = GANTrainer(img_size=IMG_SIZE, z_dim=Z_DIM, device=DEVICE)

            # Get the actual image size from checkpoint if available
            try:
                if os.path.exists(CHECKPOINT_PATH):
                    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
                    saved_img_size = checkpoint.get('img_size')
                    if saved_img_size is not None:
                        # Use the exact image size from checkpoint if available
                        if saved_img_size != IMG_SIZE:
                            print(f"[INFO] Using image size {saved_img_size} from checkpoint")
                            trainer = GANTrainer(img_size=saved_img_size, z_dim=Z_DIM, device=DEVICE)
                    # Backwards compatibility for older checkpoints without 'img_size'
                    elif checkpoint.get('G') and len(checkpoint['G']) > 0:
                        # Extract first layer weight shape to determine model size
                        for key in checkpoint['G']:
                            if 'weight' in key and len(checkpoint['G'][key].shape) > 2:
                                # Update the model size from checkpoint if needed
                                img_size_factor = checkpoint['G'][key].shape[2] // 4
                                if img_size_factor > 1:
                                    detected_size = IMG_SIZE * img_size_factor
                                    print(f"[INFO] Detected image size: {detected_size}")
                                    trainer = GANTrainer(img_size=detected_size, z_dim=Z_DIM, device=DEVICE)
                                break
            except Exception as e:
                print(f"[WARN] Failed to detect model size from checkpoint: {str(e)}")
            
            # Now load the checkpoint onto the correctly sized model
            trainer.load_checkpoint()
            img_bytes = trainer.generate_sample()
                
            # Return the image as base64
            img_str = base64.b64encode(img_bytes).decode("utf-8")
            return img_str
        except Exception as e:
            # Fallback to Keras generator if PyTorch fails
            print(f"[ERROR] PyTorch GAN failed, falling back to Keras: {str(e)}")
            noise = np.random.normal(0, 1, (1, gan.input_shape[0]))
            img_arr = gan.generate_image(noise)
            if isinstance(img_arr, list) or len(img_arr.shape) == 4:
                img_arr = img_arr[0]
            img_arr = ((img_arr + 1) * 127.5).clip(0, 255).astype(np.uint8)
            try:
                img_arr = img_arr.reshape(gan.img_shape)
            except Exception:
                flat = img_arr.flatten()
                side = int(np.ceil(np.sqrt(flat.size / 3)))
                pad = side * side * 3 - flat.size
                if pad > 0:
                    flat = np.pad(flat, (0, pad), mode='constant', constant_values=0)
                img_arr = flat.reshape((side, side, 3))
            img = Image.fromarray(img_arr)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': str(time.time())
    })

@app.route('/generate', methods=['GET'])
def generate_image():
    # Clear indicator that we're using the GAN for image generation
    print("Generating image with GAN model (not diffusion)")
    img_b64 = AxolotlImageAPI.generate_single_image()
    return jsonify({
        'image': img_b64,
        'model_type': 'GAN'  # Explicitly indicate we're using GAN not diffusion
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
