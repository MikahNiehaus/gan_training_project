#!/bin/bash
# Start script for the Axolotl GAN backend

# Set Python path to include the current directory
export PYTHONPATH=$PYTHONPATH:.

# Make the script fail fast if any command fails
set -e

echo "Starting backend service..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"

# Make data directories if they don't exist
mkdir -p data/gan_samples
echo "Created directories"

# Enhanced model check with debug logging
echo "=== Running enhanced model check ==="
# Show space available
df -h .
echo ""

# Check if model exists and create viable model if needed
bash check_model.sh

# Verify the model file
echo "=== Model files after check_model.sh ==="
ls -la data/
if [ -f "data/gan_full_model.pth" ]; then
  filesize=$(du -h "data/gan_full_model.pth" | cut -f1)
  echo "Full model exists and is $filesize"
else
  echo "WARNING: Full model file still missing!"
  # Create emergency model file
  echo "Creating emergency model file..."
  python -c '
import torch, os
os.makedirs("data", exist_ok=True)
print("Creating minimal viable model...")
class G(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = torch.nn.Linear(100, 3*32*32)
    self.act = torch.nn.Tanh()
  def forward(self, x):
    return self.act(self.l1(x)).view(-1, 3, 32, 32)
g = G()
torch.save({"G": g.state_dict(), "epoch": 0, "img_size": 32}, "data/gan_full_model.pth")
print(f"Created {os.path.getsize(\"data/gan_full_model.pth\")} byte model")
'
fi

# Get port from environment or use default
PORT=${PORT:-5000}

echo "=== Starting application on port $PORT ==="
# Start the app with gunicorn
gunicorn --workers=2 --bind=0.0.0.0:$PORT app:app --log-file -
