# compare_generation.py
# This script compares the image generation methods to verify they produce similar results

import os
import torch
import subprocess
import numpy as np
import time
from PIL import Image
import argparse
import requests
from io import BytesIO
import base64

def generate_from_command_line():
    """Generate a sample image using the train_gan.py sample command"""
    print("Generating image using train_gan.py sample command...")
    sample_cmd = "python train_gan.py sample"
    subprocess.run(sample_cmd, shell=True, check=True)
    
    # Find the generated sample file
    sample_dir = os.path.join(os.path.dirname(__file__), 'data', 'gan_samples')
    files = os.listdir(sample_dir)
    sample_files = [f for f in files if f.startswith("sample_epoch") and not f.startswith("sample_epochmanual_cpu")]
    if not sample_files:
        print("Error: Could not find generated sample file")
        return None
    
    sample_file = sorted(sample_files, key=lambda x: os.path.getmtime(os.path.join(sample_dir, x)))[-1]
    sample_path = os.path.join(sample_dir, sample_file)
    print(f"Found generated sample: {sample_path}")
    
    # Load the image
    img = Image.open(sample_path)
    return img

def generate_from_endpoint(endpoint_url="http://localhost:5000/generate"):
    """Generate a sample image using the API endpoint"""
    print(f"Generating image from endpoint: {endpoint_url}")
    try:
        response = requests.get(endpoint_url)
        response.raise_for_status()
        data = response.json()
        
        # Decode base64 image
        img_data = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as e:
        print(f"Error requesting image from endpoint: {str(e)}")
        return None

def compare_images(img1, img2):
    """Compare two images and return a similarity score"""
    if img1 is None or img2 is None:
        print("Cannot compare images, one or both are None")
        return 0
    
    # Resize to same size if needed
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    
    # Convert to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate mean squared error
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Calculate structural similarity (higher is better)
    try:
        from skimage.metrics import structural_similarity as ssim
        # Convert to grayscale for SSIM
        arr1_gray = np.mean(arr1, axis=2).astype(np.uint8)
        arr2_gray = np.mean(arr2, axis=2).astype(np.uint8)
        similarity = ssim(arr1_gray, arr2_gray)
        print(f"Image similarity (SSIM): {similarity:.4f} (higher is better)")
    except ImportError:
        # Fallback if scikit-image not available
        similarity = 1.0 - mse / 255.0**2
        print(f"Image similarity: {similarity:.4f} (higher is better)")
    
    return similarity

def main():
    parser = argparse.ArgumentParser(description="Compare image generation methods")
    parser.add_argument("--endpoint", default="http://localhost:5000/generate", 
                       help="URL of the API endpoint")
    parser.add_argument("--save", action="store_true",
                       help="Save the generated images for comparison")
    args = parser.parse_args()
    
    # Generate images using both methods
    cmd_img = generate_from_command_line()
    
    # Sleep briefly to make sure server is ready
    time.sleep(1)
    endpoint_img = generate_from_endpoint(args.endpoint)
    
    # Save images if requested
    if args.save and cmd_img and endpoint_img:
        out_dir = os.path.join(os.path.dirname(__file__), 'comparison_results')
        os.makedirs(out_dir, exist_ok=True)
        
        timestamp = int(time.time())
        cmd_img.save(os.path.join(out_dir, f"cmd_sample_{timestamp}.png"))
        endpoint_img.save(os.path.join(out_dir, f"endpoint_sample_{timestamp}.png"))
        print(f"Saved comparison images to {out_dir}")
    
    # Compare the images
    if cmd_img and endpoint_img:
        similarity = compare_images(cmd_img, endpoint_img)
        
        if similarity > 0.95:
            print("✅ The images are very similar! The endpoint is working correctly.")
        elif similarity > 0.8:
            print("⚠️ The images are somewhat similar. Check the saved images for visual comparison.")
        else:
            print("❌ The images are quite different. There may be an issue with the endpoint.")
    else:
        print("Could not compare images due to generation failure")

if __name__ == "__main__":
    main()
