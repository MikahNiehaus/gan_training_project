#!/bin/bash
# Check if the GAN checkpoint exists and create a dummy one if not

DATA_DIR="data"
CHECKPOINT_PATH="$DATA_DIR/gan_checkpoint.pth"
SAMPLE_DIR="$DATA_DIR/gan_samples"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$SAMPLE_DIR"

# Record model hash for verification
MODEL_HASH=""
if [ -f "$CHECKPOINT_PATH" ]; then
    if command -v md5sum > /dev/null 2>&1; then
        MODEL_HASH=$(md5sum "$CHECKPOINT_PATH" | cut -d ' ' -f 1)
    elif command -v md5 > /dev/null 2>&1; then
        MODEL_HASH=$(md5 -q "$CHECKPOINT_PATH")
    fi
    echo "Model found with hash: $MODEL_HASH"
fi

# Check both full model and checkpoint file
FULL_MODEL_PATH="$DATA_DIR/gan_full_model.pth"

MIN_MODEL_SIZE=1000000  # 1MB minimum size for a real model

check_model_file() {
    local file=$1
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: Required model file $file is missing. Please provide a full-size, valid model."
        exit 1
    fi
    local size=$(stat -c%s "$file" 2>/dev/null || wc -c < "$file")
    if [ "$size" -lt "$MIN_MODEL_SIZE" ]; then
        echo "❌ ERROR: Model file $file is too small ($size bytes). Refusing to use a placeholder or corrupted model."
        exit 1
    fi
    echo "✅ Model file $file exists and is $size bytes."
}

# Check both full model and checkpoint file
check_model_file "$FULL_MODEL_PATH"
check_model_file "$CHECKPOINT_PATH"

echo "Model check complete."
