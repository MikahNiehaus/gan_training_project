#!/usr/bin/env pwsh
# PowerShell script for running axolotl image generation operations

param(
    [Parameter(Position=0)]
    [ValidateSet('scrape', 'preprocess', 'train', 'train-hd', 'sample', 'sample-hd', 'batch')]
    [string]$Action,
    
    [Parameter(Position=1)]
    [int]$Count = 1
)

# Help message
function Show-Help {
    Write-Host "Axolotl Image Generation Helper" -ForegroundColor Cyan
    Write-Host "Usage: ./run_axolotl.ps1 <action> [count]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor Green
    Write-Host "  scrape       - Scrape axolotl images from the web"
    Write-Host "  preprocess   - Split images into training and test sets"
    Write-Host "  train        - Train the diffusion model at base resolution"
    Write-Host "  train-hd     - Train the diffusion model at high resolution"
    Write-Host "  sample       - Generate a single axolotl image"
    Write-Host "  sample-hd    - Generate a high-resolution axolotl image"
    Write-Host "  batch        - Generate multiple axolotl images (use count parameter)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ./run_axolotl.ps1 scrape       # Scrape images from web"
    Write-Host "  ./run_axolotl.ps1 train        # Train the model"
    Write-Host "  ./run_axolotl.ps1 batch 5      # Generate 5 axolotl images"
}

# Check if action is provided
if (-not $Action) {
    Show-Help
    exit
}

# Execute based on action
switch ($Action) {
    "scrape" {
        Write-Host "Scraping axolotl images from the web..." -ForegroundColor Cyan
        python scrape_axolotl_images.py
    }
    "preprocess" {
        Write-Host "Preprocessing images..." -ForegroundColor Cyan
        python split_train_test.py --size 64 --augment
    }
    "train" {
        Write-Host "Training diffusion model at base resolution..." -ForegroundColor Cyan
        python train_diffusion.py --resolution 1.0
    }
    "train-hd" {
        Write-Host "Training diffusion model at high resolution..." -ForegroundColor Cyan
        python train_diffusion.py --resolution 1.5
    }
    "sample" {
        Write-Host "Generating axolotl image..." -ForegroundColor Cyan
        python train_diffusion.py --mode sample --steps 100
    }
    "sample-hd" {
        Write-Host "Generating high-resolution axolotl image..." -ForegroundColor Cyan
        python train_diffusion.py --mode sample --steps 200 --resolution 1.5 --upscale 1.5
    }
    "batch" {
        Write-Host "Generating $Count axolotl images..." -ForegroundColor Cyan
        python generate_axolotl.py --count $Count --steps 100
    }
    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
        Show-Help
    }
}
