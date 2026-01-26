"""
Export Utilities for EmbeddedEarth

Handles generation of downloadable files:
- PDF Reports (Operational Briefings)
- KMZ (Google Earth Super-Overlays)
- GeoJSON (GIS Vectors)
- ZIP (Raw Data Packages)
"""

import json
import zipfile
import io
from datetime import datetime
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import os
from pathlib import Path

def generate_pdf_report(results, query, aoi_geojson=None):
    """
    Generate an operational briefing PDF report.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Title Page ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "EmbeddedEarth Intelligence Report", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Mission Summary
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Mission Summary", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Search Query: {query}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Total Hits Found: {len(results)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # --- Results Pages ---
    for i, res in enumerate(results):
        score = res.get('score', 0)
        bounds = res.get('bounds')
        img_arr = res.get('image')
        heatmap_arr = res.get('heatmap')
        
        pdf.add_page()
        
        # Header
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"Result #{i+1} - Match Score: {score:.1%}", new_x="LMARGIN", new_y="NEXT")
        
        # Coordinates
        if bounds:
            minx, miny, maxx, maxy = bounds
            center_lat = (miny + maxy) / 2
            center_lon = (minx + maxx) / 2
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, f"Location: {center_lat:.6f}, {center_lon:.6f}", new_x="LMARGIN", new_y="NEXT")
            link = f"https://www.google.com/maps/search/?api=1&query={center_lat},{center_lon}"
            pdf.set_text_color(0, 0, 255)
            pdf.cell(0, 8, "View on Google Maps", link=link, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        
        pdf.ln(5)
        
        # Images (Original vs Heatmap)
        # We need to save temp images to add them to FPDF
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save Original
            img_path = temp_path / "img.png"
            if img_arr is not None:
                Image.fromarray(img_arr).save(img_path)
                pdf.image(str(img_path), x=15, y=pdf.get_y(), w=80)
                
            # Save Heatmap (Overlay)
            if heatmap_arr is not None and img_arr is not None:
                # Re-create overlay cheaply here or assume 'image' in result MIGHT be the overlay?
                # Usually result['image'] is just RGB.
                # Let's create a visual heatmap using simple matplotlib colormap logic manually or just grayscale
                # For simplicity in this util, let's just save the raw heatmap normalized 0-255 grayscale
                
                hm_path = temp_path / "heatmap.png"
                hm_norm = (heatmap_arr - heatmap_arr.min()) / (heatmap_arr.max() - heatmap_arr.min() + 1e-6)
                hm_img = Image.fromarray((hm_norm * 255).astype(np.uint8))
                hm_img.save(hm_path)
                
                pdf.image(str(hm_path), x=110, y=pdf.get_y(), w=80)
                
            pdf.ln(85) # Space for images
            
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(90, 10, "Original Satellite Imagery", align="C")
            pdf.cell(90, 10, "AI Attention Heatmap (Grad-CAM)", align="C", new_x="LMARGIN", new_y="NEXT")
    
    return pdf.output(dest='S') # Return bytes

def generate_geojson(results):
    """
    Generate GeoJSON FeatureCollection.
    """
    features = []
    
    for i, res in enumerate(results):
        bounds = res.get('bounds')
        if not bounds: continue
        
        minx, miny, maxx, maxy = bounds
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                    [minx, miny]
                ]]
            },
            "properties": {
                "id": i + 1,
                "score": float(res.get('score', 0)),
                "rank": i + 1,
                "timestamp": datetime.now().isoformat()
            }
        }
        features.append(feature)
        
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return json.dumps(geojson, indent=2)

def generate_kmz(results):
    """
    Generate KMZ file (zipped KML + Images).
    """
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>EmbeddedEarth Results</name>
"""
    kml_footer = """</Document>\n</kml>"""
    
    mem_zip = io.BytesIO()
    
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        kml_body = ""
        
        for i, res in enumerate(results):
            img_arr = res.get('image')
            bounds = res.get('bounds')
            score = res.get('score', 0)
            
            if img_arr is None or bounds is None: continue
            
            # Save image to zip
            img_filename = f"overlay_{i}.png"
            img_bytes = io.BytesIO()
            Image.fromarray(img_arr).save(img_bytes, format="PNG")
            zf.writestr(img_filename, img_bytes.getvalue())
            
            # Add GroundOverlay to KML
            minx, miny, maxx, maxy = bounds
            
            kml_body += f"""
    <GroundOverlay>
        <name>Result #{i+1} ({score:.1%})</name>
        <Icon>
            <href>{img_filename}</href>
        </Icon>
        <LatLonBox>
            <north>{maxy}</north>
            <south>{miny}</south>
            <east>{maxx}</east>
            <west>{minx}</west>
        </LatLonBox>
    </GroundOverlay>
"""
        # Write doc.kml
        zf.writestr("doc.kml", kml_header + kml_body + kml_footer)
        
    return mem_zip.getvalue()

def generate_zip_package(results, query):
    """
    Generate RAW Data Package (Images + Metadata JSON).
    """
    mem_zip = io.BytesIO()
    
    metadata = {
        "query": query,
        "generated_at": datetime.now().isoformat(),
        "results": []
    }
    
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, res in enumerate(results):
            img_arr = res.get('image')
            score = res.get('score', 0)
            bounds = res.get('bounds')
            
            if img_arr is None: continue
            
            # Save Image
            filename = f"result_{i+1:03d}_score_{int(score*100)}.png"
            img_bytes = io.BytesIO()
            Image.fromarray(img_arr).save(img_bytes, format="PNG")
            zf.writestr(filename, img_bytes.getvalue())
            
            # Add to metadata
            metadata["results"].append({
                "filename": filename,
                "rank": i+1,
                "score": score,
                "bounds": bounds
            })
            
        # Write metadata
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        
    return mem_zip.getvalue()
