#!/usr/bin/env python3
"""
Simple CLI script to convert SVG files to high-resolution PNG format.
Requires: pip install cairosvg
"""

import argparse
import sys
from pathlib import Path

try:
    import cairosvg
except ImportError:
    print("Error: cairosvg not installed. Run: pip install cairosvg")
    sys.exit(1)


def convert_svg_to_png(svg_path, png_path=None, dpi=300):
    """Convert SVG to PNG with specified DPI."""
    svg_file = Path(svg_path)
    
    if not svg_file.exists():
        print(f"Error: {svg_path} not found")
        return False
    
    if png_path is None:
        png_path = svg_file.with_suffix('.png')
    
    try:
        cairosvg.svg2png(
            url=str(svg_file),
            write_to=str(png_path),
            dpi=dpi
        )
        print(f"Converted {svg_path} → {png_path} (DPI: {dpi})")
        return True
    except Exception as e:
        print(f"Error converting {svg_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert SVG to high-resolution PNG")
    parser.add_argument("svg_file", help="Input SVG file")
    parser.add_argument("-o", "--output", help="Output PNG file (default: same name with .png extension)")
    parser.add_argument("-d", "--dpi", type=int, default=300, help="DPI for output (default: 300)")
    
    args = parser.parse_args()
    
    success = convert_svg_to_png(args.svg_file, args.output, args.dpi)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()