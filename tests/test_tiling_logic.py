
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from pipeline.tiling import tile_image

def test_tiling_real_case():
    # Shape from user logs
    C, H, W = 6, 563, 639
    
    # Create synthetic data with a gradient so we can distinguish tiles
    # Channel 0: Horizontal gradient
    # Channel 1: Vertical gradient
    data = np.zeros((C, H, W), dtype=np.float32)
    
    # Fill with identifiable patterns
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    
    data[0] = x_grid / W
    data[1] = y_grid / H
    data[2] = (x_grid + y_grid) / (W + H)
    data += 0.1 # Ensure no absolute zeros
    
    print(f"Created synthetic image: {data.shape}")
    print(f"Stats: Min {data.min():.4f}, Max {data.max():.4f}, Mean {data.mean():.4f}")
    
    # Run tiling
    tiles = tile_image(data, tile_size=384, overlap=0.5)
    
    print(f"Generated {len(tiles)} tiles")
    
    for i, tile in enumerate(tiles):
        print(f"\nTile {i}:")
        print(f"  Coords: x={tile.x}, y={tile.y}")
        print(f"  Size: {tile.width}x{tile.height}")
        print(f"  Data Shape: {tile.data.shape}")
        print(f"  Data Stats: Min {tile.data.min():.4f}, Max {tile.data.max():.4f}, Mean {tile.data.mean():.4f}")
        
        # Verify it matches the source
        source_slice = data[:, tile.y:tile.y+tile.height, tile.x:tile.x+tile.width]
        if not np.allclose(tile.data, source_slice):
            print("  ⚠️ Data MISMATCH with source!")
        else:
            print("  ✅ Data matches source")
            
        if tile.data.max() == 0:
            print("  ⚠️ Tile is empty!")

if __name__ == "__main__":
    test_tiling_real_case()
