from flask import Flask, jsonify, send_file
import io
import base64
from models.generator import AxolotlGenerator
import numpy as np
from PIL import Image
from flask_cors import CORS
import os
import time
import subprocess
import sys

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Ensure directories exist
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# === Model integrity check at startup ===
VERIFY_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'verify_model_integrity.py')
if os.path.exists(VERIFY_SCRIPT):
    print("[STARTUP] Verifying model file integrity...")
    result = subprocess.run([sys.executable, VERIFY_SCRIPT], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[FATAL] Model file integrity check failed. Aborting startup.")
        print(result.stderr)
        sys.exit(1)
    else:
        print("[STARTUP] Model file integrity check passed.")
else:
    print(f"[WARN] Model integrity check script not found at {VERIFY_SCRIPT}. Skipping model check.")

gan = AxolotlGenerator()

class AxolotlImageAPI:
    @staticmethod
    def generate_single_image():
        logs = []
        def log(msg):
            print(msg)
            logs.append(str(msg))
        try:
            log('--- [BACKEND] Starting GAN image generation ---')
            import torch
            import torch.nn as nn
            from torchvision.utils import save_image
            from models.gan_modules import Generator, Discriminator
            import numpy as np
            log(f"torch version: {torch.__version__}")
            log(f"torchvision version: {__import__('torchvision').__version__}")
            log(f"numpy version: {np.__version__}")
            # Make sure we resolve paths correctly - this is critical for Railway deployment
            DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
            FULL_MODEL_PATH = os.path.join(DATA_DIR, 'gan_full_model.pth')
            log(f"Resolved DATA_DIR: {DATA_DIR}")
            log(f"Looking for model at: {FULL_MODEL_PATH}")
            log(f"Does model file exist? {os.path.exists(FULL_MODEL_PATH)}")
            IMG_SIZE = 32
            Z_DIM = 100
            DEVICE = torch.device('cpu')
            log(f"DEVICE: {DEVICE}")
            class GANTrainer:
                def __init__(self, img_size, z_dim, device):
                    self.img_size = img_size
                    self.z_dim = z_dim
                    self.device = device
                    self.G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
                    self.fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
                def load_model(self):
                    log(f"Checking file system details:")
                    if not os.path.exists(DATA_DIR):
                        log(f"ERROR: Data directory does not exist at: {DATA_DIR}")
                        # Try to list the parent directory to debug
                        parent_dir = os.path.dirname(DATA_DIR)
                        if os.path.exists(parent_dir):
                            log(f"Contents of parent directory {parent_dir}:")
                            for item in os.listdir(parent_dir):
                                log(f"  {item}")
                        else:
                            log(f"Parent directory {parent_dir} does not exist")
                        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}")
                        
                    log(f"Contents of data directory {DATA_DIR}:")
                    for item in os.listdir(DATA_DIR):
                        item_path = os.path.join(DATA_DIR, item)
                        size = os.path.getsize(item_path) if os.path.isfile(item_path) else "DIR"
                        log(f"  {item} - {size} bytes")
                    
                    # Define a minimum viable size for a real model file (at least 10KB)
                    MIN_MODEL_SIZE = 10 * 1024
                    
                    if os.path.exists(FULL_MODEL_PATH):
                        file_size = os.path.getsize(FULL_MODEL_PATH)
                        log(f"Loading full model from: {FULL_MODEL_PATH} (Size: {file_size} bytes)")
                        
                        if file_size < MIN_MODEL_SIZE:
                            log(f"WARNING: Full model file is suspiciously small ({file_size} bytes)")
                            log(f"This may be a placeholder file created by check_model.sh")
                            log(f"Attempting to regenerate model file using the included script...")
                            
                            # Try to regenerate a viable model file
                            try:
                                from models.gan_modules import Generator, Discriminator
                                import torch.nn as nn
                                import torch.optim as optim
                                
                                # Create minimal generator and discriminator
                                G = Generator(z_dim=self.z_dim, img_channels=3, img_size=self.img_size).to(self.device)
                                D = Discriminator(img_channels=3, img_size=self.img_size).to(self.device)
                                
                                # Save a proper model file
                                torch.save({
                                    'G': G.state_dict(),
                                    'D': D.state_dict(),
                                    'epoch': 1,
                                    'img_size': self.img_size
                                }, FULL_MODEL_PATH)
                                
                                log(f"Successfully regenerated model file: {FULL_MODEL_PATH}")
                                file_size = os.path.getsize(FULL_MODEL_PATH)
                                log(f"New file size: {file_size} bytes")
                                
                                # Now load the newly created model
                                checkpoint = torch.load(FULL_MODEL_PATH, map_location=self.device)
                                log(f"Checkpoint keys: {list(checkpoint.keys())}")
                                self.G.load_state_dict(checkpoint['G'])
                                log("Loaded regenerated model for sample generation")
                                return
                            except Exception as e:
                                log(f"ERROR: Failed to regenerate model: {str(e)}")
                                log("Continuing with attempt to load existing file...")
                        
                        try:
                            checkpoint = torch.load(FULL_MODEL_PATH, map_location=self.device)
                            log(f"Checkpoint keys: {list(checkpoint.keys())}")
                            self.G.load_state_dict(checkpoint['G'])
                            log("Loaded full model for generating sample")
                        except Exception as e:
                            log(f"ERROR: Failed to load full model: {str(e)}")
                            log("Attempting to use checkpoint file as fallback...")
                            
                            # Try using checkpoint file as fallback
                            CHECKPOINT_PATH = os.path.join(DATA_DIR, 'gan_checkpoint.pth')
                            if os.path.exists(CHECKPOINT_PATH):
                                try:
                                    log(f"Trying checkpoint file: {CHECKPOINT_PATH}")
                                    checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
                                    self.G.load_state_dict(checkpoint['G'])
                                    log("Successfully loaded checkpoint as fallback")
                                    return
                                except Exception as e2:
                                    log(f"ERROR: Failed to load checkpoint: {str(e2)}")
                            
                            # If we get here, both attempts failed
                            log("ERROR: All model load attempts failed")
                            raise Exception(f"Failed to load any model file: {str(e)}")
                    else:
                        log("ERROR: Full model file (gan_full_model.pth) not found.")
                        log(f"Full model path expected at: {FULL_MODEL_PATH}")
                        log("This file is created automatically every 1000 epochs during training.")
                        log("Make sure train_gan.py has run for at least 1000 epochs.")
                        raise FileNotFoundError("Full model file (gan_full_model.pth) not found.")
                def generate_sample(self):
                    log("Calling G.eval() and generating image...")
                    self.G.eval()
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise[:1]).detach().cpu()
                        log(f"Fake image tensor shape: {fake.shape}")
                        buffer = io.BytesIO()
                        save_image(fake, buffer, format="PNG", normalize=True)
                        buffer.seek(0)
                        log("Image saved to buffer and ready to encode.")
                        return buffer.getvalue()
            trainer = GANTrainer(img_size=IMG_SIZE, z_dim=Z_DIM, device=DEVICE)
            trainer.load_model()
            img_bytes = trainer.generate_sample()
            log("Image generation complete, encoding to base64...")
            img_str = base64.b64encode(img_bytes).decode("utf-8")
            log("Image base64 encoding complete.")
            return img_str, logs
        except Exception as e:
            log(f"PyTorch GAN failed: {str(e)}")
            return None, logs

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': str(time.time())
    })

@app.route('/generate', methods=['GET'])
def generate_image():
    img_b64, logs = AxolotlImageAPI.generate_single_image()
    return jsonify({'image': img_b64, 'logs': logs})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)