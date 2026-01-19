import torch
import torch.nn as nn
import os
from huggingface_hub import hf_hub_download
import numpy as np
from typing import Dict, Any, List, Optional

# Import the reconstructed model architecture
try:
    from .copernicus.model_vit import vit_base_patch16
except ImportError:
    # Fallback for when running directly or in different path structure
    from models.copernicus.model_vit import vit_base_patch16

class CopernicusFM(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        print("Loading CopernicusFM model...")
        # Initialize the model architecture
        # global_pool=False to use CLS token (likely match for foundation model weights)
        model = vit_base_patch16(
            num_classes=0,
            drop_rate=0.0,
            global_pool=True, 
            loc_option='lonlat'
        )

        # Download weights from Hugging Face
        repo_id = "wangyi111/Copernicus-FM"
        filename = "CopernicusFM_ViT_base_varlang_e100.pth"
        
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Helper to remove prefix if needed (though usually not for this repo)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # Handle potential key mismatches or formatting
            # clean_state_dict = {}
            # for k, v in state_dict.items():
            #     # standard cleaning if needed
            #     clean_state_dict[k] = v
                
            msg = model.load_state_dict(state_dict, strict=False)
            
            # Check for benign warning
            is_benign = (
                len(msg.unexpected_keys) == 0 and 
                set(msg.missing_keys) == {'norm.weight', 'norm.bias'}
            )
            
            if is_benign:
                print("CopernicusFM model loaded successfully (ignoring expected missing normalization keys).")
            else:
                print(f"Model loaded with msg: {msg}")
            
        except Exception as e:
            print(f"Error loading CopernicusFM weights: {e}")
            raise e

        return model

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Standard normalization if required by the model
        # The demo code didn't show explicit normalization other than what the data loader might do.
        # Assuming input is [B, C, H, W] in 0-1 range or tailored raw values.
        # DOFA usually expects normalized inputs.
        return x

    def forward(self, 
                x: torch.Tensor, 
                meta_info: torch.Tensor, 
                wavelengths: List[float], 
                bandwidths: List[float],
                input_mode: str = 'spectral') -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            meta_info: Tensor [B, 4] containing [lon, lat, time, area]
            wavelengths: List of central wavelengths in nm
            bandwidths: List of bandwidths in nm
            input_mode: 'spectral' or 'variable'
        """
        
        # Ensure inputs are on correct device
        x = x.to(self.device)
        meta_info = meta_info.to(self.device)
        
        # Prepare wavelengths and bandwidths
        # model expects list or tensor? forward_features convets to tensor.
        # forward signature: forward(self, x, meta_info, wave_list, bandwidth, language_embed, input_mode, kernel_size=None)
        
        # Dummy language embed for spectral mode
        language_embed = torch.zeros(1).to(self.device) # Should be ignored
        
        with torch.no_grad():
            output = self.model(
                x, 
                meta_info, 
                wavelengths, 
                bandwidths, 
                language_embed, 
                input_mode
            )
            
            # CopernicusFMViT returns (x, fx) tuple
            # We want x (the embedding)
            if isinstance(output, tuple):
                features = output[0]
            else:
                features = output
            
        return features

    def extract_features(self, x: torch.Tensor, meta_info: torch.Tensor, wavelengths: List[float], bandwidths: List[float]) -> torch.Tensor:
        return self.forward(x, meta_info, wavelengths, bandwidths, input_mode='spectral')
