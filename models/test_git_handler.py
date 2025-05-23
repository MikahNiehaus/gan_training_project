#!/usr/bin/env python
"""
Test script for Git model handler
This script allows you to test the Git model handler functionality
without having to run a full training process.
"""

import os
import sys
import argparse

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Import the GitModelHandler
from models.git_model_handler import GitModelHandler

def test_git_handler(model_path, branch='main'):
    """
    Test the Git model handler with a specific model file
    
    Args:
        model_path: Path to the model file
        branch: Git branch to use
    """
    print(f"Testing Git model handler with model: {model_path}")
    print(f"Target branch: {branch}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist: {model_path}")
        return False
        
    # Create the handler
    print("Creating Git model handler...")
    handler = GitModelHandler(model_path, branch=branch)
    
    # Test Git repo check
    print("\nChecking if we're in a Git repository...")
    is_repo = handler._is_git_repo()
    print(f"In Git repository: {is_repo}")
    if not is_repo:
        print("Not in a Git repository. Test failed.")
        return False
    
    # Test if file is tracked
    print("\nChecking if model file is tracked...")
    is_tracked = handler._is_file_tracked()
    print(f"Model file is tracked: {is_tracked}")
    
    if not is_tracked:
        print("\nAdding model file to Git...")
        added = handler.add_model_file()
        print(f"Model file added: {added}")
    
    # Test commit (but don't actually commit)
    print("\nSimulating commit (no actual commit)...")
    print(f"Would commit with message: 'Test commit for model at {model_path}'")
    
    # Test push (but don't actually push)
    print("\nSimulating push to remote (no actual push)...")
    print(f"Would push to {handler.remote}/{handler.branch}")
    
    print("\nTest completed successfully!")
    print("\nTo perform an actual Git push, run:")
    print(f"python -c \"from models.git_model_handler import GitModelHandler; GitModelHandler('{model_path}').update_model_in_git(epoch_num=0)\"")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Git model handler")
    parser.add_argument("--model", type=str, default="data/gan_checkpoint.pth", 
                        help="Path to model file (relative to backend directory)")
    parser.add_argument("--branch", type=str, default="main", 
                        help="Git branch to use")
    args = parser.parse_args()
    
    # Convert relative path to absolute path
    model_path = os.path.join(backend_dir, args.model)
    
    test_git_handler(model_path, args.branch)
