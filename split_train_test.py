import os
import glob
import shutil
import random
import argparse
import hashlib
from PIL import Image
import time

# Paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'data', 'axolotl_scraped')
TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'data', 'train')
TEST_DIR = os.path.join(os.path.dirname(__file__), 'data', 'test')

# Parameters
TEST_SPLIT = 0.2  # 20% for test
TARGET_SIZE = (64, 64)  # Default size - can be overridden by command line

# Function to compute image hash for duplicate detection
def image_hash(file_path):
    """Return a hash of the image file for duplicate detection."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

# Function to check if an image is a duplicate
def is_duplicate(image_hash, existing_hashes):
    """Check if image has already been seen."""
    return image_hash in existing_hashes

# Function to resize and validate an image
def resize_and_validate(src_path, crop_to_square=True):
    """Validate image and optionally crop to square, but do NOT resize."""
    try:
        img = Image.open(src_path).convert('RGB')
        # Check if image is too small or malformed
        if min(img.size) < 32:  # Arbitrary minimum size for sanity
            print(f"Skipping small image: {src_path}, size={img.size}")
            return None
        # Crop to square if needed (center crop)
        if crop_to_square:
            w, h = img.size
            if w != h:
                size = min(w, h)
                left = (w - size) // 2
                top = (h - size) // 2
                right = left + size
                bottom = top + size
                img = img.crop((left, top, right, bottom))
        return img
    except Exception as e:
        print(f"Failed to process {src_path}: {e}")
        return None

def main(source_dir=DATASET_DIR, train_dir=TRAIN_DIR, test_dir=TEST_DIR,
         target_size=TARGET_SIZE, test_split=TEST_SPLIT, augment=False):
    """Process the images and split into train/test sets, preserving original quality."""
    print(f"Processing images from: {source_dir}")
    print(f"Images will be cropped to square but NOT resized. Original quality preserved.")
    
    # Make sure we have necessary directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all image files
    image_files = glob.glob(os.path.join(source_dir, '*.*'))
    print(f"Found {len(image_files)} source images")
    
    # Track duplicates to avoid having same image in train/test
    processed_hashes = set()
    valid_files = []
    
    # First pass: validate images and collect unique ones
    for file_path in image_files:
        img_hash = image_hash(file_path)
        
        # Skip if duplicate
        if img_hash in processed_hashes:
            continue
            
        # Add to processed set
        processed_hashes.add(img_hash)
        valid_files.append(file_path)
    
    # Shuffle and split
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * (1 - test_split))
    train_files = valid_files[:split_idx]
    test_files = valid_files[split_idx:]
    
    print(f"Processing {len(train_files)} train images and {len(test_files)} test images")
    
    # Process and copy train files
    train_count = 0
    for src_file in train_files:
        img = resize_and_validate(src_file, crop_to_square=True)
        if img is not None:
            timestamp = int(time.time() * 1000)
            filename = f"train_{train_count}_{timestamp}.jpg"
            img.save(os.path.join(train_dir, filename), quality=95)
            train_count += 1
            
            # Basic data augmentation if requested
            if augment and train_count % 5 == 0:  # Augment ~20% of images
                # Flip horizontally
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                filename = f"train_aug_{train_count}_{timestamp}.jpg"
                flipped.save(os.path.join(train_dir, filename), quality=95)
                train_count += 1
    
    # Process and copy test files
    test_count = 0
    for src_file in test_files:
        img = resize_and_validate(src_file, crop_to_square=True)
        if img is not None:
            timestamp = int(time.time() * 1000)
            filename = f"test_{test_count}_{timestamp}.jpg"
            img.save(os.path.join(test_dir, filename), quality=95)
            test_count += 1
    
    print(f"Successfully processed {train_count} training images and {test_count} test images")
    print(f"All images cropped to square, original resolution preserved.")
    
    return train_count, test_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and split axolotl dataset')
    parser.add_argument('--size', type=int, default=64, 
                      help='(Ignored) Kept for compatibility. Images are NOT resized, only cropped to square.')
    parser.add_argument('--source', type=str, default=DATASET_DIR,
                      help='Source directory containing images')
    parser.add_argument('--train', type=str, default=TRAIN_DIR,
                      help='Output directory for training set')
    parser.add_argument('--test', type=str, default=TEST_DIR,
                      help='Output directory for test set')
    parser.add_argument('--split', type=float, default=TEST_SPLIT,
                      help='Portion of data to use for testing (0.0-1.0)')
    parser.add_argument('--augment', action='store_true',
                      help='Apply basic data augmentation')
                      
    args = parser.parse_args()
    
    # Run with parsed arguments
    main(
        source_dir=args.source,
        train_dir=args.train,
        test_dir=args.test,
        target_size=(args.size, args.size),  # Kept for compatibility, not used
        test_split=args.split,
        augment=args.augment
    )
