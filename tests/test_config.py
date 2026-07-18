"""
Tests for config.py

Validates that singleton config objects have physically correct and
internally consistent values. No model loading or GEE required.
"""

import pytest
from config import (
    sentinel2_bands, sentinel1_bands, model_config,
    tiling_config, search_config, gee_config,
)


# ── ModelConfig ────────────────────────────────────────────────────────────────

class TestModelConfig:
    def test_model_name_points_to_geolb(self):
        """Bug-fix regression: must use the GeoLB model, not XShadow/DOFA-CLIP."""
        assert "earthflow/GeoLB" in model_config.model_name, (
            "model_name should point to the GeoLB hub model, "
            f"got: {model_config.model_name!r}"
        )

    def test_model_name_is_hub_format(self):
        """Model name must be in hf-hub: format for open_clip."""
        assert model_config.model_name.startswith("hf-hub:"), (
            f"model_name should start with 'hf-hub:', got: {model_config.model_name!r}"
        )

    def test_embedding_dim_is_1152(self):
        """GeoLB ViT-SO400M outputs 1152-dim embeddings, not 768."""
        assert model_config.embedding_dim == 1152

    def test_image_size_is_384(self):
        """GeoLB is a 384px SigLIP model."""
        assert model_config.image_size == 384

    def test_patch_size_is_14(self):
        assert model_config.patch_size == 14

    def test_image_produces_valid_patch_grid(self):
        """
        GeoLB uses SigLIP-So400m with 384px and patch_size=14.
        384 / 14 = 27.4 → floor to 27 patches per side (729 total).
        The model handles non-divisible sizes via floor-patching.
        This test just confirms the grid is at least 1×1.
        """
        import math
        patches_per_side = math.floor(model_config.image_size / model_config.patch_size)
        assert patches_per_side >= 1


# ── Sentinel2Bands ─────────────────────────────────────────────────────────────

class TestSentinel2Bands:
    def test_band_names_are_gee_style(self):
        """Config must use GEE band names (B2, not B02)."""
        expected = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        assert sentinel2_bands.band_names == expected

    def test_scale_factor(self):
        """Sentinel-2 L2A reflectance is stored as integer×10000."""
        assert sentinel2_bands.scale_factor == 10000.0

    def test_wavelengths_all_bands_present(self):
        """Every band in band_names must have a wavelength entry."""
        for band in sentinel2_bands.band_names:
            assert band in sentinel2_bands.wavelengths, f"Missing wavelength for {band}"

    def test_bandwidths_all_bands_present(self):
        """Every band in band_names must have a bandwidth entry."""
        for band in sentinel2_bands.band_names:
            assert band in sentinel2_bands.bandwidths, f"Missing bandwidth for {band}"

    def test_wavelengths_are_optical_nm(self):
        """All S2 wavelengths must be in the visible-to-SWIR optical range (nm)."""
        for band, wl in sentinel2_bands.wavelengths.items():
            assert 400 <= wl <= 2500, f"Band {band} wavelength {wl} nm out of optical range"

    def test_wavelengths_in_nm_not_um(self):
        """Values must be in nanometers, not micrometers (min > 400)."""
        for band, wl in sentinel2_bands.wavelengths.items():
            assert wl > 100, f"Band {band} wavelength {wl} looks like μm, expected nm"

    def test_get_wavelength_tensor_length(self):
        wls = sentinel2_bands.get_wavelength_tensor()
        assert len(wls) == len(sentinel2_bands.band_names)

    def test_get_wavelength_tensor_order(self):
        """Order must match band_names order."""
        wls = sentinel2_bands.get_wavelength_tensor()
        for i, band in enumerate(sentinel2_bands.band_names):
            assert wls[i] == sentinel2_bands.wavelengths[band]

    def test_get_bandwidth_list_length(self):
        bws = sentinel2_bands.get_bandwidth_list()
        assert len(bws) == len(sentinel2_bands.band_names)

    def test_bandwidths_positive(self):
        for band, bw in sentinel2_bands.bandwidths.items():
            assert bw > 0, f"Band {band} bandwidth must be positive"


# ── Sentinel1Bands ─────────────────────────────────────────────────────────────

class TestSentinel1Bands:
    def test_band_names(self):
        assert 'VV' in sentinel1_bands.band_names
        assert 'VH' in sentinel1_bands.band_names

    def test_wavelengths_in_microns(self):
        """S1 C-band is ~5.55 cm = 55500 μm. Must be stored in μm."""
        for band, wl in sentinel1_bands.wavelengths.items():
            assert 50000 < wl < 60000, (
                f"S1 band {band} wavelength {wl} should be ~55500 μm (C-band)"
            )

    def test_get_wavelength_tensor_matches_band_names(self):
        wls = sentinel1_bands.get_wavelength_tensor()
        assert len(wls) == len(sentinel1_bands.band_names)


# ── TilingConfig ───────────────────────────────────────────────────────────────

class TestTilingConfig:
    def test_tile_size(self):
        assert tiling_config.tile_size == 384

    def test_overlap_ratio(self):
        assert 0.0 <= tiling_config.overlap_ratio < 1.0

    def test_stride_is_positive(self):
        assert tiling_config.stride > 0

    def test_stride_formula(self):
        expected = int(tiling_config.tile_size * (1 - tiling_config.overlap_ratio))
        assert tiling_config.stride == expected


# ── SearchConfig ───────────────────────────────────────────────────────────────

class TestSearchConfig:
    def test_similarity_threshold_range(self):
        assert 0.0 <= search_config.similarity_threshold <= 1.0

    def test_top_k_positive(self):
        assert search_config.top_k > 0


# ── GEEConfig ──────────────────────────────────────────────────────────────────

class TestGEEConfig:
    def test_s2_collection_name(self):
        assert "S2_SR" in gee_config.s2_collection

    def test_cloud_threshold_range(self):
        assert 0.0 < gee_config.cloud_threshold < 1.0

    def test_max_cloud_cover_range(self):
        assert 0 <= gee_config.max_cloud_cover <= 100
