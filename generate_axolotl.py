#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Axolotl Images - Robust image generation script
This script simplifies the process of generating axolotl images
using our trained diffusion model, with robust error handling.
"""

import os
import sys
import argparse
import torch
from PIL import Image
from datetime import datetime

# Import from train_diffusion.py
try:
    from train_diffusion import SimpleUNet, sample_image, print_gpu_memory_status, clean_memory
except ImportError:
    print("Error: Could not import from train_diffusion.py")
    print("Make sure you're running this script from the backend directory")
    sys.exit(1)

def generate_axolotls(count=1, output_dir=None, model_path=None, use_cpu=False, steps=100, 
                  resolution_scale=None, upscale_factor=1.0, debug=False):
    """
    Generate multiple axolotl images with robust error handling
    
    Args:
        count: Number of images to generate
        output_dir: Directory to save images
        model_path: Path to model weights
        use_cpu: Force CPU usage
        steps: Number of diffusion steps
        resolution_scale: Override model's resolution scale
        upscale_factor: Further upscaling factor to apply to output
        debug: Enable debug logging
    """
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'generated_axolotls')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return
    
    # Default model path
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'data', 'best_diffusion_model.pth')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Start generation
    print(f"Generating {count} axolotl images...")
    print(f"Using {'CPU' if use_cpu else 'GPU if available'} with {steps} diffusion steps")
    
    if debug:
        print_gpu_memory_status()
    
    # Generate images
    successful = 0
    for i in range(count):
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"axolotl_{timestamp}_{i}.png")
        
        print(f"\nGenerating image {i+1}/{count}...")
        
        # Clean memory before each generation
        clean_memory()
        
        try:
            # Generate image
            result_path = sample_image(
                model_path=model_path,
                out_path=output_path,
                steps=steps,
                use_cpu_fallback=use_cpu,
                resolution_scale=resolution_scale,
                upscale_factor=upscale_factor
            )
            if result_path:
                successful += 1
                print(f"Successfully generated image: {result_path}")
            else:
                print(f"Failed to generate image {i+1}")
            
            # Display memory status in debug mode
            if debug and torch.cuda.is_available():
                print_gpu_memory_status()
                
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            
            # Switch to CPU if GPU is causing problems
            if not use_cpu and "CUDA" in str(e):
                print("CUDA error detected, switching to CPU for remaining images...")
                use_cpu = True
    
    # Print summary
    print(f"\nGeneration complete: {successful}/{count} images generated successfully")
    print(f"Images saved to: {output_dir}")
    
    return successful

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Axolotl Images with Diffusion Model')
    parser.add_argument('--count', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--steps', type=int, default=100, help='Number of diffusion steps')
    parser.add_argument('--resolution', type=float, default=None, 
                        help='Resolution scale factor (1.0=original, 2.0=double)')
    parser.add_argument('--upscale', type=float, default=1.0,
                        help='Additional upscaling factor for final image')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Print system info
    print("=" * 60)
    print("Axolotl Image Generator")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
          # Generate images
    try:
        generate_axolotls(
            count=args.count,
            output_dir=args.output,
            model_path=args.model,
            use_cpu=args.cpu,
            steps=args.steps,
            resolution_scale=args.resolution,
            upscale_factor=args.upscale,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
