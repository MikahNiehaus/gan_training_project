"""
Git Model Handler - A class for managing model file versioning with Git
"""
import os
import subprocess
import time
import logging
from datetime import datetime

class GitModelHandler:
    """
    Class to handle Git operations for model files
    
    This class provides OOP methods for pushing model changes to Git,
    specifically designed for automatically versioning ML model checkpoints.
    """
    
    def __init__(self, model_path, branch='main', remote='origin'):
        """
        Initialize the Git model handler
        
        Args:
            model_path: Path to the model file
            branch: Git branch to push to (default: 'main')
            remote: Git remote to push to (default: 'origin')
        """
        # Auto-detect correct model path
        abs_path = os.path.abspath(model_path)
        if not os.path.exists(abs_path):
            # Try relative to backend dir if not found
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_path = os.path.abspath(os.path.join(backend_dir, model_path))
            if os.path.exists(alt_path):
                abs_path = alt_path
        self.model_path = abs_path
        self.branch = branch
        self.remote = remote
        self.last_push_time = 0
        self.cooldown_seconds = 10  # Minimum time between pushes
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("GitModelHandler")
    
    def _run_git_command(self, command):
        """
        Run a Git command and handle the output
        
        Args:
            command: List containing the command and its arguments
            
        Returns:
            Tuple of (success, output)
        """
        try:
            # Add git to the beginning of the command
            full_command = ["git"] + command
            self.logger.debug(f"Running: {' '.join(full_command)}")
            
            # Run the command and capture output
            result = subprocess.run(
                full_command, 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode != 0:
                self.logger.error(f"Git command failed: {result.stderr.strip()}")
                return False, result.stderr.strip()
            
            return True, result.stdout.strip()
        
        except Exception as e:
            self.logger.error(f"Git command failed with exception: {str(e)}")
            return False, str(e)
    
    def _is_git_repo(self):
        """Check if we're inside a git repository"""
        success, _ = self._run_git_command(["rev-parse", "--is-inside-work-tree"])
        return success
    
    def _is_file_tracked(self):
        """Check if the model file is tracked by Git"""
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model file does not exist: {self.model_path}")
            return False
            
        success, _ = self._run_git_command(["ls-files", "--error-unmatch", self.model_path])
        return success
    
    def add_model_file(self):
        """Add the model file to Git and ensure LFS tracking"""
        if not os.path.exists(self.model_path):
            self.logger.error(f"Cannot add non-existent file: {self.model_path}")
            return False
        
        # Check file size to ensure it's not empty
        file_size = os.path.getsize(self.model_path)
        if file_size == 0:
            self.logger.error(f"Cannot add empty file (0 bytes): {self.model_path}")
            return False
            
        self.logger.info(f"Adding model file: {self.model_path} ({file_size:,} bytes)")
        
        # Ensure Git LFS is tracking this file type
        ext = os.path.splitext(self.model_path)[1]
        if ext in ['.pth', '.pt', '.h5', '.bin', '.onnx']:
            lfs_track_cmd = ["lfs", "track", f"*{ext}"]
            self.logger.info(f"Ensuring Git LFS is tracking '*{ext}'...")
            self._run_git_command(lfs_track_cmd)
            # Always add .gitattributes if changed
            self._run_git_command(["add", ".gitattributes"])
            
        success, output = self._run_git_command(["add", self.model_path])
        if success:
            self.logger.info(f"Added model file to Git staging: {self.model_path}")
        return success
    
    def commit_model_file(self, message=None):
        """
        Commit the model file
        
        Args:
            message: Custom commit message, or auto-generated if None
        """
        if not message:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"Update model checkpoint at {timestamp}"
            
        success, output = self._run_git_command(["commit", "-m", message, self.model_path])
        if success:
            self.logger.info(f"Committed model update: {message}")
        else:
            # Check if there were no changes
            if "nothing to commit" in output:
                self.logger.info("No changes in model file to commit")
                # This is still considered a success for our workflow
                return True
                
        return success
    
    def push_model_update(self):
        """Push the model update to the remote repository"""
        success, output = self._run_git_command(["push", self.remote, self.branch])
        if success:
            self.logger.info(f"Pushed model update to {self.remote}/{self.branch}")
        return success
    
    def update_model_in_git(self, epoch_num=None):
        """
        Main method to update a model file in Git.

        This method handles the full process: add, commit, and push.
        It is designed to be called automatically during training to version model checkpoints.

        Args:
            epoch_num: Optional epoch number to include in commit message.

        Returns:
            Boolean indicating success.

        Steps:
            1. Rate limiting: Prevents too frequent pushes (default cooldown: 10s).
            2. Checks if inside a git repository.
            3. Prepares a commit message (optionally with epoch number).
            4. Adds the model file to git staging.
            5. Commits the model file (skips if no changes).
            6. Pushes the commit to the remote branch.
            7. Updates last push time if successful.
        """
        # Rate limiting to prevent too many pushes
        current_time = time.time()
        if current_time - self.last_push_time < self.cooldown_seconds:
            self.logger.info(f"Skipping push - cooldown period active ({self.cooldown_seconds}s)")
            return False
        # Check if in a git repository
        if not self._is_git_repo():
            self.logger.error("Not in a git repository")
            return False
        # Prepare the commit message
        message = "Update model checkpoint"
        if epoch_num is not None:
            message += f" at epoch {epoch_num}"
        # Add and commit
        if not self.add_model_file():
            return False
        if not self.commit_model_file(message):
            return False
        # Push the changes
        push_success = self.push_model_update()
        if push_success:
            self.last_push_time = current_time
        return push_success
