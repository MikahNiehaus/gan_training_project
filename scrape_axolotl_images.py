import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from PIL import Image
import re
import pytesseract
import hashlib
import time
import shutil
import random

def has_watermark_text(file_path):
    try:
        img = Image.open(file_path)
        # Convert to grayscale for OCR
        gray = img.convert('L')
        # Run OCR
        text = pytesseract.image_to_string(gray)
        # If text is detected and not just whitespace, likely a watermark
        if text.strip():
            return True
        return False
    except Exception:
        return False

def is_valid_image(file_path):
    # Filter out images with watermarks, text, or non-axolotl content (basic heuristics)
    try:
        img = Image.open(file_path)
        # Only keep images that are reasonably large and RGB
        if img.mode != 'RGB' or min(img.size) < 128:
            return False
        # Watermark detection
        if has_watermark_text(file_path):
            return False
        # Optionally, add more advanced checks here
        return True
    except Exception:
        return False

def remove_invalid_images(folder):
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not is_valid_image(fpath):
            os.remove(fpath)

def image_hash(file_path):
    """Return a hash of the image file for duplicate detection."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def get_existing_hashes(folder):
    """Get a set of hashes for all images in the folder."""
    hashes = set()
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        h = image_hash(fpath)
        if h:
            hashes.add(h)
    return hashes

if __name__ == '__main__':
    OUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'axolotl_scraped')
    os.makedirs(OUT_DIR, exist_ok=True)

    # Remove old train and test folders before scraping
    train_dir = os.path.join(os.path.dirname(__file__), 'data', 'train')
    test_dir = os.path.join(os.path.dirname(__file__), 'data', 'test')
    for d in [train_dir, test_dir]:
        if os.path.exists(d):
            print(f"Deleting old directory: {d}")
            shutil.rmtree(d)

    bing_crawler = BingImageCrawler(storage={'root_dir': OUT_DIR})

    # Large, diverse list of search queries
    keywords = [
    "real axolotl photo", "authentic axolotl photograph", "close-up photo of real axolotl",
    "macro shot of live axolotl", "photo of axolotl in aquarium", "realistic axolotl underwater",
    "photograph of live axolotl", "axolotl in natural habitat", "high-res axolotl photo",
    "unedited axolotl image", "real axolotl macro closeup", "axolotl in clean aquarium",
    "documentary style axolotl image", "realistic axolotl pet photo", "photo of axolotl with natural lighting",
    "real axolotl face close-up", "axolotl gills detailed photo", "natural axolotl skin texture",
    "realistic axolotl head photo", "axolotl in glass tank real photo", "raw image of axolotl swimming",
    "realistic axolotl body macro", "real axolotl dorsal view photo", "photo of real albino axolotl",
    "real axolotl on gravel", "realistic axolotl among plants", "axolotl in stream real photo",
    "axolotl eating worm photo", "real axolotl foot close-up", "live axolotl in aquarium HD photo",
    "true photo of axolotl in water", "realistic axolotl eyes close-up", "real axolotl breathing underwater",
    "clear water axolotl photography", "real pet axolotl captured on camera", "real-life axolotl swimming photo"
]



    used_keywords = set()
    total_downloaded = len(os.listdir(OUT_DIR))
    existing_hashes = get_existing_hashes(OUT_DIR)
    while total_downloaded < 5000 and len(used_keywords) < len(keywords):
        for kw in keywords:
            if kw in used_keywords:
                continue
            print(f"Searching for: {kw}")
            # Download directly to OUT_DIR
            before_files = set(os.listdir(OUT_DIR))
            bing_crawler.crawl(
                keyword=kw,
                filters={
                    'type': 'photo',
                    'size': 'large',
                    'color': 'color'
                },
                max_num=100000000
            )
            # Check for new files and remove duplicates
            after_files = set(os.listdir(OUT_DIR))
            new_files = after_files - before_files
            for fname in new_files:
                fpath = os.path.join(OUT_DIR, fname)
                h = image_hash(fpath)
                # Only remove the file if it is a duplicate or invalid, never touch pre-existing files
                if not h or h in existing_hashes or not is_valid_image(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
                else:
                    existing_hashes.add(h)
            used_keywords.add(kw)
            total_downloaded = len(os.listdir(OUT_DIR))
            print(f"Total images so far: {total_downloaded}")
            if total_downloaded >= 5000:
                break
            time.sleep(2)
    print(f"Scraping complete. Images saved to {OUT_DIR}")

    # --- Split scraped images into train and test sets ---
    all_images = [f for f in os.listdir(OUT_DIR) if os.path.isfile(os.path.join(OUT_DIR, f))]
    random.shuffle(all_images)
    split_idx = int(0.9 * len(all_images))
    train_imgs = all_images[:split_idx]
    test_imgs = all_images[split_idx:]
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for fname in train_imgs:
        shutil.copy2(os.path.join(OUT_DIR, fname), os.path.join(train_dir, fname))
    for fname in test_imgs:
        shutil.copy2(os.path.join(OUT_DIR, fname), os.path.join(test_dir, fname))
    print(f"Split complete: {len(train_imgs)} train, {len(test_imgs)} test images.")
