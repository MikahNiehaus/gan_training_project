"""
Fix for GAN gradient checkpointing to improve memory usage and fix the grey images issue.
"""
import os
import torch
import torch.nn as nn

def enable_gradient_checkpointing(model, level=1):
    """
    Enable gradient checkpointing to save VRAM memory at the cost of computation speed.
    
    Args:
        model: The PyTorch model (Generator or Discriminator)
        level: Aggressiveness level (1-3) for memory savings
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import needed for gradient checkpointing
        import torch.utils.checkpoint as checkpoint
        
        # For models with gradient_checkpointing_enable method
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            return True
            
        # Sequential modules (most common case)
        if isinstance(model, nn.Sequential):
            # Store original forward method
            if not hasattr(model, '_original_forward'):
                model._original_forward = model.forward
            
            # Create checkpointed forward method
            def checkpointed_forward(self, x):
                modules = list(self.children())
                if len(modules) == 0:
                    return self._original_forward(x)
                
                # For level 1, checkpoint the whole sequential
                if level == 1:
                    return checkpoint.checkpoint(self._original_forward, x)
                
                # For levels 2-3, break it into chunks for finer-grained checkpointing
                chunk_size = 2 if level >= 2 else 3  # Smaller chunks for higher levels
                chunks = [modules[i:i+chunk_size] for i in range(0, len(modules), chunk_size)]
                result = x
                for chunk in chunks:
                    seq = nn.Sequential(*chunk)
                    result = checkpoint.checkpoint(seq, result)
                return result
            
            # Bind the method to the model
            import types
            model.forward = types.MethodType(checkpointed_forward, model)
            return True
            
        # For any other module type, recurse through its children
        for name, child in model.named_children():
            # Skip if this is a primitive module with no children
            if len(list(child.children())) == 0:
                continue
                
            # Apply to children
            if isinstance(child, (nn.Sequential, nn.ModuleList)) or len(list(child.children())) > 0:
                enable_gradient_checkpointing(child, level)
        
        # Save memory with other techniques based on level
        if level >= 2:
            # Make sure model is using inplace operations where possible
            for m in model.modules():
                if isinstance(m, (nn.ReLU, nn.LeakyReLU)):
                    m.inplace = True
        
        return True
    except Exception as e:
        print(f"[VRAM] Error enabling gradient checkpointing: {str(e)}")
        return False

def disable_gradient_checkpointing(model):
    """
    Disable gradient checkpointing and restore normal forward pass
    
    Args:
        model: The PyTorch model (Generator or Discriminator)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # For models with gradient_checkpointing_disable method
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
            return True
            
        # Restore original forward method for Sequential modules
        if isinstance(model, nn.Sequential) and hasattr(model, '_original_forward'):
            model.forward = model._original_forward
            delattr(model, '_original_forward')
            return True
            
        # Recursively handle children
        for name, child in model.named_children():
            if len(list(child.children())) == 0:
                continue
                
            if isinstance(child, (nn.Sequential, nn.ModuleList)) or len(list(child.children())) > 0:
                disable_gradient_checkpointing(child)
                
        return True
    except Exception as e:
        print(f"[VRAM] Error disabling gradient checkpointing: {str(e)}")
        return False
