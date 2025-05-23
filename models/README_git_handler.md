# Git Model Handler for ML Checkpoints

This module provides automated Git version control for machine learning model checkpoints.

## Overview

The `GitModelHandler` class offers an object-oriented approach to automatically push model checkpoints to Git at specified intervals during training. This ensures you never lose important model weights and can track the evolution of your model over time.

## Features

- **Automated Model Versioning**: Automatically commits and pushes model file changes
- **Epoch Tracking**: Includes epoch numbers in commit messages
- **Rate Limiting**: Prevents excessive Git operations with built-in cooldown
- **Error Handling**: Gracefully manages Git errors without interrupting training
- **OOP Design**: Clean class-based implementation

## Usage

The handler is automatically integrated with the GAN training process:

- Every 100 epochs: Model checkpoint is saved locally
- Every 1000 epochs: Model checkpoint is pushed to Git main branch

## Requirements

- Git must be installed and accessible in the PATH
- Training must run in a valid Git repository
- You must have push access to the repository's main branch

## Configuration

You can modify the Git push interval in `train_gan.py` by changing the `git_push_interval` property.

## Manual Git Operations

If needed, you can also use the PowerShell script to manually track models:

```powershell
# Run this PowerShell script to ensure model changes are tracked
.\track_models.ps1
```
