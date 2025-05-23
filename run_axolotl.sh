#!/bin/bash
# Shell script for running axolotl image generation operations

# Help message
show_help() {
    echo -e "\e[36mAxolotl Image Generation Helper\e[0m"
    echo -e "\e[33mUsage: ./run_axolotl.sh <action> [count]\e[0m"
    echo ""
    echo -e "\e[32mActions:\e[0m"
    echo "  scrape       - Scrape axolotl images from the web"
    echo "  preprocess   - Split images into training and test sets"
    echo "  train        - Train the diffusion model at base resolution"
    echo "  train-hd     - Train the diffusion model at high resolution"
    echo "  sample       - Generate a single axolotl image"
    echo "  sample-hd    - Generate a high-resolution axolotl image"
    echo "  batch        - Generate multiple axolotl images (use count parameter)"
    echo ""
    echo -e "\e[32mExamples:\e[0m"
    echo "  ./run_axolotl.sh scrape       # Scrape images from web"
    echo "  ./run_axolotl.sh train        # Train the model"
    echo "  ./run_axolotl.sh batch 5      # Generate 5 axolotl images"
}

# Check if action is provided
if [ -z "$1" ]; then
    show_help
    exit 1
fi

# Set count parameter (default to 1)
COUNT=${2:-1}

# Execute based on action
case "$1" in
    "scrape")
        echo -e "\e[36mScraping axolotl images from the web...\e[0m"
        python scrape_axolotl_images.py
        ;;
    "preprocess")
        echo -e "\e[36mPreprocessing images...\e[0m"
        python split_train_test.py --size 64 --augment
        ;;
    "train")
        echo -e "\e[36mTraining diffusion model at base resolution...\e[0m"
        python train_diffusion.py --resolution 1.0
        ;;
    "train-hd")
        echo -e "\e[36mTraining diffusion model at high resolution...\e[0m"
        python train_diffusion.py --resolution 1.5
        ;;
    "sample")
        echo -e "\e[36mGenerating axolotl image...\e[0m"
        python train_diffusion.py --mode sample --steps 100
        ;;
    "sample-hd")
        echo -e "\e[36mGenerating high-resolution axolotl image...\e[0m"
        python train_diffusion.py --mode sample --steps 200 --resolution 1.5 --upscale 1.5
        ;;
    "batch")
        echo -e "\e[36mGenerating $COUNT axolotl images...\e[0m"
        python generate_axolotl.py --count $COUNT --steps 100
        ;;
    *)
        echo -e "\e[31mUnknown action: $1\e[0m"
        show_help
        exit 1
        ;;
esac
