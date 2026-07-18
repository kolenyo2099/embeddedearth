"""
Grad-ECLIP Implementation

Provides gradient-based explainability for DOFA-CLIP models,
specifically adapted for Vision Transformers as specified in research.md.
"""

import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import model_config

class GradECLIP:
    """
    Grad-ECLIP: Gradient-based Explanation for CLIP.
    
    Adapted for the custom DOFACLIPWrapper architecture.
    """
    
    def __init__(
        self,
        model_wrapper,
        target_layer_name: str = "vision_model.trunk.norm"
    ):
        """
        Initialize Grad-ECLIP.

        Args:
            model_wrapper: Instance of DOFACLIPWrapper.
            target_layer_name: Dot-path to the target layer, traversed from
                model_wrapper. 'vision_model.trunk.norm' resolves to
                model_wrapper.vision_model.trunk.norm, i.e. the final LayerNorm
                of the ViT trunk before global pooling.
        """
        self.wrapper = model_wrapper
        self.model = model_wrapper  # formatting alias
        self.target_layer_name = target_layer_name
        
        # Storage
        self._activations = None
        self._gradients = None
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        layer = self._get_layer(self.target_layer_name)

        if layer is None:
            print(f"Error: GradECLIP could not find layer '{self.target_layer_name}'. "
                  "Heatmaps will be blank.")
            return
             
        # print(f"GradECLIP: Hooking into {layer}")
        
        # Hook for capturing activations (forward)
        h1 = layer.register_forward_hook(self._forward_hook)
        
        # Hook for capturing gradients (backward)
        h2 = layer.register_full_backward_hook(self._backward_hook)
        
        self.hooks.append(h1)
        self.hooks.append(h2)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _get_layer(self, layer_path: str):
        """Traverse the model to find the layer."""
        parts = layer_path.split('.')
        obj = self.wrapper
        
        for part in parts:
            if part.startswith('-'):
                try:
                    idx = int(part)
                    if hasattr(obj, '__getitem__'):
                        obj = obj[idx]
                    elif hasattr(obj, 'children'):
                        obj = list(obj.children())[idx]
                    else:
                        return None
                except:
                    return None
            else:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return None
        return obj

    def _forward_hook(self, module, input, output):
        """Capture activations (output of the layer)."""
        self._activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients w.r.t. output."""
        self._gradients = grad_output[0]

    def generate_gradcam(
        self,
        image: torch.Tensor,
        text: str,
        image_size: int = 384,
        wavelengths: torch.Tensor = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM Heatmap.

        Uses gradients to weight the contribution of each visual token feature channel
        to the final similarity score.

        Args:
            image: Preprocessed image tensor (B, C, H, W) in [0, 1] range.
            text: Text query string.
            image_size: Output heatmap resolution.
            wavelengths: Band wavelengths in μm (must match the sensor used for
                         embedding). Defaults to Sentinel-2 if None.
        """
        # 1. Clear previous hooks state
        self._activations = None
        self._gradients = None
        self.model.model.zero_grad()

        # 2. Forward Pass: Encode Image
        with torch.set_grad_enabled(True):
            image_tensor = self.wrapper.preprocess_tensor(image, normalize=True)
            if image_tensor.requires_grad is False:
                image_tensor.requires_grad = True

            # Resolve wavelengths — use caller-supplied value (correct sensor) or fall back
            if wavelengths is not None:
                wvs = wavelengths.to(self.model.device)
            else:
                from config import sentinel2_bands
                wvs = (
                    torch.tensor(sentinel2_bands.get_wavelength_tensor())
                    .float()
                    .to(self.model.device)
                    / 1000.0  # nm → μm
                )
            
            # Forward Visual TRUNK specifically (to bypass projection head issues and ensure DOFA-CLIP path)
            # Expects (B, C, H, W) -> (Embedding, Intermediates) or Embedding
            out = self.model.model.visual.trunk(image_tensor, wvs)
            if isinstance(out, tuple):
                image_emb = out[0]
            else:
                image_emb = out
                
            # Normalize embedding
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            
            # Encode Text
            text_emb = self.wrapper.encode_text(text, normalize=True).detach()
            
            # 3. Compute Similarity (Score)
            score = (image_emb * text_emb).sum()
            
            # 4. Backward Pass
            score.backward()
            
            # 5. Get Activations and Gradients from Hooks
            activations = self._activations
            gradients = self._gradients
            
            if activations is None or gradients is None:
                # print("Error: Hooks did not capture data. Check target layer name.")
                return np.zeros((image_size, image_size))
            
            # 6. Compute Weights (Global Average Pooling of Gradients over spatial dims)
            weights = torch.mean(gradients, dim=1, keepdim=True) # (1, 1, D)
            
            # 7. Weighted Sum
            cam = (weights * activations).sum(dim=2) # (1, N)
            
            # 8. ReLU
            cam = F.relu(cam)
            
            # 9. Reshape and Normalize with Robust Logic
            num_tokens = cam.shape[1]
            
            # Try to infer grid size from model config if possible
            grid_size = None
            if hasattr(self.model.model.visual.trunk, 'patch_embed'):
                 if hasattr(self.model.model.visual.trunk.patch_embed, 'grid_size'):
                      g_h, g_w = self.model.model.visual.trunk.patch_embed.grid_size
                      if isinstance(g_h, int): grid_size = g_h # assume square
                      
            # Fallback deduction
            if grid_size is None:
                 grid_size = int(np.sqrt(num_tokens))
            
            # Check for perfect square
            if grid_size * grid_size == num_tokens:
                 # Perfect fit
                 cam = cam.reshape(1, 1, grid_size, grid_size)
            else:
                 # Try removing CLS/Registers (usually at start or end?)
                 # Standard ViT: CLS at 0
                 # SigLIP: usually no CLS, just GAP.
                 # DOFA: uses standard ViT trunk usually.
                 
                 # Logic: find largest square S*S <= num_tokens
                 s = int(np.sqrt(num_tokens))
                 
                 if s * s == num_tokens - 1:
                      # One extra token (CLS)
                      cam = cam[:, 1:] # Drop first
                      cam = cam.reshape(1, 1, s, s)
                 elif s * s < num_tokens:
                      # More extra tokens (registers?)
                      # Or CLS + Distillation?
                      # Assume Spatial tokens are LAST S*S tokens
                      diff = num_tokens - (s*s)
                      cam = cam[:, diff:]
                      cam = cam.reshape(1, 1, s, s)
                 else:
                      # S*S > num_tokens? Impossible unless s was ceil, but we used int().
                      return np.zeros((image_size, image_size))
            
            # Upsample
            cam = F.interpolate(cam, size=(image_size, image_size), mode='bilinear', align_corners=False)
            
            # Normalize (0-1) for visualization
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
                
            return cam.squeeze().detach().cpu().numpy()

def generate_explanation(
    model_wrapper,
    image: np.ndarray,
    text: str,
    device: str = 'cpu',
    image_size: int = 384,
    wavelengths: torch.Tensor = None
) -> np.ndarray:
    """
    High-level explanation function using Grad-CAM.

    Args:
        model_wrapper: DOFACLIPWrapper instance.
        image: Image array in [0, 1] range, shape (C, H, W) or (H, W, C).
        text: Text query.
        device: Computation device.
        image_size: Output heatmap resolution.
        wavelengths: Band wavelengths in μm matching the sensor used for embedding.
                     Pass image_encoder.wavelengths to guarantee consistency.
                     Defaults to Sentinel-2 if None.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] <= 13:  # (H, W, C) → (C, H, W)
            image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device or model_config.device)
    image.requires_grad = True

    explainer = GradECLIP(model_wrapper)

    try:
        heatmap = explainer.generate_gradcam(
            image, text, image_size=image_size, wavelengths=wavelengths
        )
    finally:
        explainer.remove_hooks()

    return heatmap
