"""
Enhanced gradient checkpointing for GAN training to fix the gray image issue
and memory problems at higher resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for any PyTorch model.
    This helps significantly reduce memory usage during backpropagation
    at the cost of some extra computation.
    
    Args:
        model: The PyTorch model
    
    Returns:
        True if successful, False otherwise
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        # Use built-in method if available
        return model.gradient_checkpointing_enable()
    
    try:
        # For sequential modules, apply checkpoint to children
        if isinstance(model, nn.Sequential):
            # Store original forward
            model._original_forward = model.forward
            
            # Create new forward function using checkpointing
            def checkpointed_forward(self, x):
                children = list(self.children())
                if len(children) == 0:  # No children to checkpoint
                    return self._original_forward(x)
                
                # Split the sequential module into chunks of size 2-3 for optimal memory/speed tradeoff
                chunk_size = min(3, len(children))
                if len(children) <= chunk_size:
                    # If only a few children, checkpoint the whole thing
                    return checkpoint.checkpoint(self._original_forward, x)
                
                # Process in chunks for better memory efficiency
                chunks = [children[i:i+chunk_size] for i in range(0, len(children), chunk_size)]
                out = x
                for chunk in chunks:
                    # Create a temporary sequential module
                    temp_module = nn.Sequential(*chunk)
                    # Apply checkpointing to this chunk
                    out = checkpoint.checkpoint(temp_module, out)
                return out
                
            # Bind the new method to the instance
            import types
            model.forward = types.MethodType(checkpointed_forward, model)
            
        # For module lists, apply checkpoint to each module
        elif isinstance(model, nn.ModuleList):
            for i, module in enumerate(model):
                enable_gradient_checkpointing(module)
                
        # For all other modules, apply checkpointing to children modules
        else:
            # Apply to all direct children modules
            for name, module in list(model.named_children()):
                if isinstance(module, (nn.Sequential, nn.ModuleList)) or len(list(module.children())) > 0:
                    enable_gradient_checkpointing(module)
                    
        return True
    except Exception as e:
        print(f"[VRAM] Error enabling gradient checkpointing: {str(e)}")
        return False

def disable_gradient_checkpointing(model):
    """
    Disable gradient checkpointing for a PyTorch model.
    This restores the original forward pass.
    
    Args:
        model: The PyTorch model
    
    Returns:
        True if successful, False otherwise
    """
    if hasattr(model, 'gradient_checkpointing_disable'):
        # Use built-in method if available
        return model.gradient_checkpointing_disable()
    
    try:
        # For sequential modules, restore original forward
        if isinstance(model, nn.Sequential):
            if hasattr(model, '_original_forward'):
                model.forward = model._original_forward
                delattr(model, '_original_forward')
                
        # For module lists, disable checkpoint to each module
        elif isinstance(model, nn.ModuleList):
            for i, module in enumerate(model):
                disable_gradient_checkpointing(module)
                
        # For all other modules, disable checkpointing from children modules
        else:
            # Apply to all direct children modules
            for name, module in list(model.named_children()):
                if isinstance(module, (nn.Sequential, nn.ModuleList)) or len(list(module.children())) > 0:
                    disable_gradient_checkpointing(module)
                    
        return True
    except Exception as e:
        print(f"[VRAM] Error disabling gradient checkpointing: {str(e)}")
        return False
