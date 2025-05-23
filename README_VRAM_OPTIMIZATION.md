# Axolotl Image Generator - VRAM Optimization Update

This update implements gradient checkpointing and VRAM optimization for the Axolotl Image Generator GAN, ensuring it can run efficiently on systems with limited GPU memory while still producing high-quality images.

## Key Improvements

### 1. Gradient Checkpointing Implementation

The GAN models now use PyTorch's gradient checkpointing mechanism to drastically reduce VRAM usage during training:

- Split Generator and Discriminator into separate layers that can be individually checkpointed
- Dynamically enable/disable checkpointing based on VRAM needs
- Uses `torch.utils.checkpoint` to save memory during training by trading computation for memory

### 2. Multi-level VRAM Optimization

Three levels of optimization are now available, depending on VRAM pressure:

- **Level 1:** Basic gradient checkpointing
- **Level 2:** Reduced batch size + gradient checkpointing
- **Level 3:** Mixed precision training + reduced batch size + gradient checkpointing

### 3. Automatic Fallback Mechanisms

The system now automatically detects CUDA out-of-memory errors and recovers:

- Detects VRAM limitations during training and automatically enables increasingly aggressive optimizations
- Falls back to CPU processing when necessary for generation
- Provides detailed logging of VRAM optimization steps

### 4. Enhanced Checkpoint Management

The model checkpoint saving and loading has been improved:

- Safely disables gradient checkpointing during checkpoint saving to ensure clean state
- Properly re-enables optimization features after loading checkpoints
- Better error handling for checkpoint loading failures

### 5. Git Integration

Git model versioning continues to work seamlessly with the new VRAM optimizations:

- Automatically handles pushing model updates to Git every 1000 epochs
- Properly handles binary model files without consuming excessive Git storage

## Testing

A new testing script `test_vram_optimization.py` has been added that:

- Verifies gradient checkpointing implementation works correctly
- Tests sample generation with various VRAM constraints
- Can simulate VRAM pressure to test fallback mechanisms

Run the test with:

```bash
python test_vram_optimization.py
```

For testing under VRAM pressure:

```bash
python test_vram_optimization.py --pressure
```

## Usage Notes

- **Training:** No changes needed - the system will automatically adapt to VRAM constraints
- **Generation:** The API endpoint remains the same and will automatically use optimizations if needed
- **Deployment:** The Railway deployment configuration works with these optimizations

## VRAM Requirements

With these optimizations, the system can now train and generate:
- **Minimum:** 2GB VRAM (with aggressive optimizations)
- **Recommended:** 4GB VRAM or more
- **CPU Fallback:** Works on systems with no GPU, though much slower
