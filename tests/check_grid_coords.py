
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.tiling import generate_geo_grid
import numpy as np

def check_grid(lat, lon, name):
    print(f"--- Checking Grid around {name} ({lat}, {lon}) ---")
    # Define a small AOI around the point (approx 10km box)
    delta = 0.05
    bounds = (lon - delta, lat - delta, lon + delta, lat + delta)
    
    # Generate grid
    tiles = list(generate_geo_grid(bounds, resolution=10.0))
    print(f"Generated {len(tiles)} tiles.")
    
    # Print first few tiles
    for i, (t_bounds, col, row) in enumerate(tiles[:5]):
        minx, miny, maxx, maxy = t_bounds
        w_deg = maxx - minx
        h_deg = maxy - miny
        
        # Center
        cy = (miny + maxy) / 2
        cx = (minx + maxx) / 2
        
        # Approximate meters
        h_m = h_deg * 111320
        w_m = w_deg * 111320 * np.cos(np.radians(cy))
        
        print(f"Tile {i} (Col {col}, Row {row}):")
        print(f"  Bounds: Lon [{minx:.5f}, {maxx:.5f}], Lat [{miny:.5f}, {maxy:.5f}]")
        print(f"  Center: {cy:.5f}, {cx:.5f}")
        print(f"  Size (deg): {w_deg:.5f} x {h_deg:.5f}")
        print(f"  Size (m):   {w_m:.1f} x {h_m:.1f} (Target 3840.0)")
        
        # Check alignment
        if i == 0:
            # First tile should be at Top-Left of AOI
            # AOI Top-Left is (lon-delta, lat+delta)
            # Tile should start near there.
            print(f"  AOI Top-Left: {lat+delta:.5f}, {lon-delta:.5f}")
            print(f"  Tile Top-Left: {maxy:.5f}, {minx:.5f}")
            
            lat_diff = abs((lat+delta) - maxy)
            lon_diff = abs((lon-delta) - minx)
            print(f"  Diff: Lat {lat_diff:.6f}, Lon {lon_diff:.6f}")

check_grid(40.4168, -3.7038, "Madrid")
