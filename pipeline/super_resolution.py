
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.rrdbnet import RRDBNet
from config import model_config

class SuperResModel:
    """
    Wrapper for Real-ESRGAN Super-Resolution Model.
    Upscales Sentinel-2 imagery (RGB) by 4x.
    """
    
    def __init__(self, weights_path: str = "models/weights/RealESRGAN_x4plus.pth", device: str = None):
        self.device = device or model_config.device
        self.model_path = Path(project_root) / weights_path
        self._model = None
        
        # RRDBNet standard config for RealESRGAN x4 plus
        self.model_args = {
            'in_nc': 3,
            'out_nc': 3,
            'nf': 64,
            'nb': 23,
            'gc': 32
        }

    def load(self):
        """Load the model weights."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"SR weights not found at {self.model_path}. Please run download script.")
            
        print(f"Loading SR Model from {self.model_path}...")
        
        # Initialize architecture
        model = RRDBNet(**self.model_args)

        # Load weights from disk
        loadnet = torch.load(str(self.model_path), map_location=self.device)
        
        # Prepare state dict
        if 'params_ema' in loadnet:
            state_dict = loadnet['params_ema']
        elif 'params' in loadnet:
            state_dict = loadnet['params']
        else:
            state_dict = loadnet

        # --- KEY CONVERSION FIX ---
        # The downloaded RealESRGAN weights use different key names than our RRDBNet definition.
        # We assume the architecture is identical, just named differently.
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            # 1. specific outer layer mapping (ORDER MATTERS)
            if 'conv_body' in new_k:
                new_k = new_k.replace('conv_body', 'trunk_conv')
            elif 'conv_up1' in new_k:
                new_k = new_k.replace('conv_up1', 'upconv1')
            elif 'conv_up2' in new_k:
                new_k = new_k.replace('conv_up2', 'upconv2')
            elif 'conv_hr' in new_k:
                new_k = new_k.replace('conv_hr', 'HRconv')
            
            # 2. Main body mapping
            # Only do this if we haven't already mapped header/footer layers
            # Note: We replaced 'conv_body' already so 'body' substring won't match trunk_conv
            if 'body.' in new_k:
                new_k = new_k.replace('body.', 'RRDB_trunk.')
            
            # 3. RDB Block Case Sensitivity (rdb -> RDB)
            if 'rdb' in new_k:
                new_k = new_k.replace('rdb', 'RDB')
            
            new_state_dict[new_k] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        model.to(self.device)
        
        self._model = model
        print("SR Model loaded successfully.")

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale an RGB image by 4x.
        
        Args:
            image: Numpy array (C, H, W) or (H, W, C), float 0-1 or uint8 0-255.
                   If float, assumed (C, H, W).
                   If uint8, assumed (H, W, C).
                   
        Returns:
            Upscaled image (C, H*4, W*4) as float32 0-1.
        """
        if self._model is None:
            self.load()
            
        # Preprocess
        is_uint8 = image.dtype == np.uint8
        
        if is_uint8:
            # (H, W, C) -> (C, H, W) float 0-1
            img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            # Assumed (C, H, W) float 0-1
            img_tensor = torch.from_numpy(image).float()
            
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0) # (1, C, H, W)
            
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self._model(img_tensor)
            
        # Postprocess
        output = output.squeeze(0).cpu().numpy() # (C, H*4, W*4)
        output = np.clip(output, 0, 1)
        
        return output

# Singleton
_sr_instance = None

def get_sr_model():
    global _sr_instance
    if _sr_instance is None:
        _sr_instance = SuperResModel()
    return _sr_instance
