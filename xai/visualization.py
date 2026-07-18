"""
Visualization Module

Generates heatmap overlays and result visualizations
for the Streamlit frontend.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: RGB image (H, W, 3) in uint8.
        heatmap: Heatmap array (H, W) in [0, 1].
        alpha: Overlay transparency.
        colormap: Matplotlib colormap name.
        
    Returns:
        Blended image (H, W, 3) in uint8.
    """
    # Ensure image is RGB
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Resize heatmap to match image
    h, w = image.shape[:2]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize((w, h), Image.BICUBIC)) / 255.0
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Remove alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend
    blended = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    
    return blended


def create_colorbar(
    colormap: str = 'jet',
    width: int = 20,
    height: int = 200
) -> np.ndarray:
    """
    Create a standalone colorbar image.
    
    Args:
        colormap: Matplotlib colormap name.
        width: Colorbar width.
        height: Colorbar height.
        
    Returns:
        Colorbar image (height, width, 3).
    """
    cmap = plt.get_cmap(colormap)
    
    gradient = np.linspace(1, 0, height).reshape(-1, 1)
    gradient = np.tile(gradient, (1, width))
    
    colorbar = cmap(gradient)[:, :, :3]
    colorbar = (colorbar * 255).astype(np.uint8)
    
    return colorbar


def generate_caption(
    score: float,
    bounds: tuple = None,
    index: int = None
) -> str:
    """
    Generate accessible caption for a result tile.
    
    Args:
        score: Similarity score.
        bounds: Geospatial bounds (minx, miny, maxx, maxy).
        index: Tile index.
        
    Returns:
        Descriptive caption string.
    """
    parts = []
    
    if index is not None:
        parts.append(f"Result {index + 1}")
    
    parts.append(f"Similarity: {score:.1%}")
    
    if bounds:
        minx, miny, maxx, maxy = bounds
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2
        ns = "N" if center_lat >= 0 else "S"
        ew = "E" if center_lon >= 0 else "W"
        parts.append(f"Location: {abs(center_lat):.4f}°{ns}, {abs(center_lon):.4f}°{ew}")

    return ". ".join(parts)


def save_figure_to_bytes(fig) -> bytes:
    """
    Convert matplotlib figure to bytes.
    
    Args:
        fig: Matplotlib figure.
        
    Returns:
        PNG bytes.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return buf.getvalue()


def create_comparison_figure(
    original: np.ndarray,
    heatmap_overlay: np.ndarray,
    title: str = None
) -> bytes:
    """
    Create side-by-side comparison of original and heatmap.
    
    Args:
        original: Original RGB image.
        heatmap_overlay: Image with heatmap overlay.
        title: Optional title.
        
    Returns:
        PNG bytes of the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_overlay)
    axes[1].set_title("Explanation Heatmap")
    axes[1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    result = save_figure_to_bytes(fig)
    plt.close(fig)
    
    return result
