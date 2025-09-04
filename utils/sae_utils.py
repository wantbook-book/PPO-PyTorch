import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from torch import Tensor

from sae_lens import SAE

class GlobalSAE:
    use_sae = True

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


# 全局变量存储激活数据
class ActivationLogger:
    def __init__(self):
        self.before_activations = []
        self.after_activations = []
        self.enabled = False
    
    def clear(self):
        self.before_activations.clear()
        self.after_activations.clear()
    
    def enable(self):
        self.enabled = True
        self.clear()
    
    def disable(self):
        self.enabled = False
    
    def enable_logging(self):
        """Enable activation logging and clear existing data"""
        self.enable()
    
    def disable_logging(self):
        """Disable activation logging"""
        self.disable()
    
    def has_data(self):
        """Check if there is any recorded activation data"""
        return len(self.before_activations) > 0 or len(self.after_activations) > 0
    
    def get_data(self):
        """Get all recorded activation data with metadata"""
        import datetime
        return {
            'activations_before': self.before_activations,
            'activations_after': self.after_activations,
            'metadata': {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_samples': len(self.before_activations),
                'data_format': 'numpy_float32'
            }
        }
    
    def clear_data(self):
        """Clear all recorded activation data"""
        self.clear()

activation_logger = ActivationLogger()

def get_intervention_hook(
    sae: SAE,
    feature_idx: int,
    max_activation: float = 1.0,
    strength: float = 1.0,
    min_strength: float = 0.0,
    max_strength: float = 1.0,
):
    def hook_fn(module, input, output):
        if not GlobalSAE.use_sae:
            return output

        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()

        if sae.device != activations.device:
            sae.device = activations.device
            sae.to(sae.device)

        # 记录激活前的数据
        if activation_logger.enabled:
            activation_logger.before_activations.append(activations.detach().cpu().float().numpy())

        features = sae.encode(activations)
        reconstructed = sae.decode(features)
        error = activations.to(features.dtype) - reconstructed

        features[..., feature_idx] = max_activation * strength

        activations_hat = sae.decode(features) + error
        activations_hat = activations_hat.type_as(activations)

        # 记录激活后的数据
        if activation_logger.enabled:
            activation_logger.after_activations.append(activations_hat.detach().cpu().float().numpy())

        if torch.is_tensor(output):
            return activations_hat
        else:
            return (activations_hat,) + output[1:] if len(output) > 1 else (activations_hat,)

    return hook_fn


def get_multi_intervention_hook(
    sae: SAE,
    feature_idxs: list[int],
    max_activations: list[float],
    strengths: list[float],
    min_strength: float = 0.0,
    max_strength: float = 1.0,
):
    def hook_fn(module, input, output):
        if not GlobalSAE.use_sae:
            return output

        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()

        if sae.device != activations.device:
            sae.device = activations.device
            sae.to(sae.device)

        features = sae.encode(activations)
        reconstructed = sae.decode(features)
        error = activations.to(features.dtype) - reconstructed

        for feature_idx, max_activation, strength in zip(feature_idxs, max_activations, strengths):
            features[..., feature_idx] = max_activation * strength

        activations_hat = sae.decode(features) + error
        activations_hat = activations_hat.type_as(activations)

        if torch.is_tensor(output):
            return activations_hat
        else:
            return (activations_hat,) + output[1:] if len(output) > 1 else (activations_hat,)

    return hook_fn

def get_clamp_hook(
    direction: Tensor,
    max_activation: float = 1.0,
    strength: float = 1.0,
    min_strength: float = 0.0,
    max_strength: float = 1.0,
):
    def hook_fn(module, input, output):
        if not GlobalSAE.use_sae:
            return output
        
        nonlocal direction
        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()
        
        direction = direction / torch.norm(direction)
        direction = direction.type_as(activations)
        proj_magnitude = torch.sum(activations * direction, dim=-1, keepdim=True)
        orthogonal_component = activations - proj_magnitude * direction

        clamped = orthogonal_component + direction * max_activation * strength

        if torch.is_tensor(output):
            return clamped
        else:
            return (clamped,) + output[1:] if len(output) > 1 else (clamped,)

    return hook_fn
