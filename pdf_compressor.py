#!/usr/bin/env python3
"""
PDF Compressor CLI Tool
Compresses PDF files to a target size with minimal quality loss.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import tempfile
import shutil

try:
    from pdf2image import convert_from_path
    from PIL import Image
    import img2pdf
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# Optional OCR support
try:
    import pytesseract
    from PyPDF2 import PdfWriter, PdfReader
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import inch
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def parse_size(size_str: str) -> int:
    """Parse size string (e.g., '5MB', '500KB') to bytes."""
    size_str = size_str.upper().strip()
    
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    
    for unit, multiplier in multipliers.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[:-len(unit)])
                return int(value * multiplier)
            except ValueError:
                pass
    
    # Try parsing as plain number (assume bytes)
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}. Use format like '5MB', '500KB', etc.")


def compress_pdf(
    input_path: str,
    output_path: str,
    target_size: Optional[int] = None,
    quality: int = 85,
    dpi: int = 200,
    verbose: bool = False
) -> bool:
    """
    Compress PDF by converting to images and back.
    
    Args:
        input_path: Path to input PDF
        output_path: Path to output PDF
        target_size: Target size in bytes (None for no target)
        quality: JPEG quality (1-100, higher = better quality)
        dpi: DPI for image conversion (lower = smaller file)
        verbose: Whether to print progress messages
    
    Returns:
        True if compression successful, False otherwise
    """
    try:
        # Convert PDF to images
        if verbose:
            tqdm.write(f"Converting PDF to images at {dpi} DPI...")
        images = convert_from_path(input_path, dpi=dpi)
        
        if not images:
            if verbose:
                print("Error: Could not extract images from PDF")
            return False
        
        # Compress images
        compressed_images = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create progress bar
            progress_bar = tqdm(
                enumerate(images),
                total=len(images),
                desc="Compressing pages",
                unit="page",
                disable=not verbose,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            for i, image in progress_bar:
                # Convert to RGB if necessary (for JPEG compatibility)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # For very aggressive compression, resize if image is very large
                # This helps when DPI alone isn't enough
                max_dimension = 2000  # Maximum width or height in pixels
                if quality < 30 and (image.width > max_dimension or image.height > max_dimension):
                    # Calculate new dimensions maintaining aspect ratio
                    ratio = min(max_dimension / image.width, max_dimension / image.height)
                    new_width = int(image.width * ratio)
                    new_height = int(image.height * ratio)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save as JPEG with specified quality and optimization
                temp_path = os.path.join(temp_dir, f"page_{i}.jpg")
                # Use progressive JPEG for better compression at lower quality
                save_kwargs = {
                    'quality': quality,
                    'optimize': True,
                }
                if quality < 50:
                    save_kwargs['progressive'] = True
                
                image.save(temp_path, 'JPEG', **save_kwargs)
                compressed_images.append(temp_path)
            
            # Convert images back to PDF
            if verbose:
                tqdm.write("Converting images back to PDF...")
            with open(output_path, 'wb') as f:
                f.write(img2pdf.convert(compressed_images))
            
            return True
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        if verbose:
            print(f"Error during compression: {e}")
            import traceback
            traceback.print_exc()
            return False


def create_searchable_pdf(
    input_path: str,
    output_path: str,
    dpi: int = 300,
    verbose: bool = False
) -> bool:
    """
    Create an editable PDF from scanned PDF using OCR.
    
    Args:
        input_path: Path to input PDF
        output_path: Path to output PDF
        dpi: DPI for image conversion (higher = better OCR accuracy)
        verbose: Whether to print progress messages
    
    Returns:
        True if successful, False otherwise
    """
    if not OCR_AVAILABLE:
        print("Error: OCR dependencies not installed. Install with: pip install pytesseract PyPDF2 reportlab")
        return False
    
    try:
        if verbose:
            tqdm.write(f"Converting PDF to images at {dpi} DPI for OCR...")
        images = convert_from_path(input_path, dpi=dpi)
        
        if not images:
            if verbose:
                print("Error: Could not extract images from PDF")
            return False
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create PDF with editable text using reportlab
            c = canvas.Canvas(output_path)
            
            # Create progress bar
            progress_bar = tqdm(
                enumerate(images),
                total=len(images),
                desc="OCR processing",
                unit="page",
                disable=not verbose,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            for i, image in progress_bar:
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Get image dimensions
                img_width, img_height = image.size
                
                # Determine page size (use image dimensions or default to A4)
                # Convert pixels to points (1 inch = 72 points, and we know DPI)
                page_width = (img_width / dpi) * 72
                page_height = (img_height / dpi) * 72
                
                # Set page size
                c.setPageSize((page_width, page_height))
                
                # Save image temporarily
                temp_img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                image.save(temp_img_path, 'JPEG', quality=95)
                
                # Draw image as background
                c.drawImage(temp_img_path, 0, 0, width=page_width, height=page_height, preserveAspectRatio=True)
                
                # Get OCR data with detailed information
                # Use pytesseract to get text with bounding boxes
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng')
                
                # Group text by line for better text extraction
                # Extract text with positions
                n_boxes = len(ocr_data['text'])
                text_objects = []
                
                for j in range(n_boxes):
                    text = ocr_data['text'][j].strip()
                    if text and len(text) > 0:  # Only process non-empty text
                        # Get bounding box coordinates
                        x = ocr_data['left'][j]
                        y = ocr_data['top'][j]
                        w = ocr_data['width'][j]
                        h = ocr_data['height'][j]
                        conf = int(ocr_data['conf'][j])
                        
                        # Only add text with reasonable confidence
                        if conf > 30:  # Confidence threshold
                            text_objects.append({
                                'text': text,
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h,
                                'conf': conf
                            })
                
                # Add text objects to PDF
                # Sort by Y coordinate (top to bottom) and X coordinate (left to right)
                text_objects.sort(key=lambda t: (-t['y'], t['x']))
                
                for text_obj in text_objects:
                    # Convert image coordinates to PDF coordinates
                    # PDF coordinates: (0,0) is bottom-left, image coordinates: (0,0) is top-left
                    pdf_x = (text_obj['x'] / dpi) * 72
                    pdf_y = page_height - ((text_obj['y'] + text_obj['h']) / dpi) * 72  # Flip Y coordinate
                    
                    # Calculate font size to match text height
                    font_size = max(8, min(72, (text_obj['h'] / dpi) * 72 * 0.9))
                    
                    # Add text as editable text object
                    # Use very transparent text so it's selectable/editable but doesn't obscure the image
                    # The text will be fully editable in PDF editors like Adobe Acrobat
                    c.setFillColorRGB(0, 0, 0)  # Black color
                    c.setFont("Helvetica", font_size)
                    
                    # Use very low alpha so text is there but barely visible
                    # This ensures text is selectable, copyable, and editable
                    c.setFillAlpha(0.01)  # Almost invisible but still creates text object
                    c.drawString(pdf_x, pdf_y, text_obj['text'])
                    c.setFillAlpha(1.0)  # Reset alpha
                
                # Start new page
                c.showPage()
            
            # Save PDF
            c.save()
            
            if verbose:
                tqdm.write("✓ OCR completed successfully! PDF is now editable.")
            
            return True
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        if verbose:
            print(f"\nError during OCR: {e}")
            import traceback
            traceback.print_exc()
        return False


def compress_pdf_with_ocr(
    input_path: str,
    output_path: str,
    target_size: Optional[int] = None,
    quality: int = 85,
    dpi: int = 200,
    verbose: bool = False
) -> bool:
    """
    Compress PDF and make it searchable with OCR.
    
    Args:
        input_path: Path to input PDF
        output_path: Path to output PDF
        target_size: Target size in bytes (None for no target)
        quality: JPEG quality (1-100, higher = better quality)
        dpi: DPI for image conversion
        verbose: Whether to print progress messages
    
    Returns:
        True if successful, False otherwise
    """
    if not OCR_AVAILABLE:
        print("Error: OCR dependencies not installed. Install with: pip install pytesseract PyPDF2 reportlab")
        return False
    
    try:
        # Use the specified DPI for both compression and OCR
        # Lower DPI = smaller file, but may reduce OCR accuracy
        # For aggressive compression, we allow lower DPI even if OCR accuracy suffers
        ocr_dpi = dpi
        
        if verbose:
            if dpi < 200:
                tqdm.write(f"⚠ Warning: Using low DPI ({dpi}) may reduce OCR accuracy")
            tqdm.write(f"Converting PDF to images at {ocr_dpi} DPI for compression and OCR...")
        images = convert_from_path(input_path, dpi=ocr_dpi)
        
        if not images:
            if verbose:
                print("Error: Could not extract images from PDF")
            return False
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create PDF with compressed images and editable text using reportlab
            c = canvas.Canvas(output_path)
            
            # Create progress bar
            progress_bar = tqdm(
                enumerate(images),
                total=len(images),
                desc="Compressing & OCR",
                unit="page",
                disable=not verbose,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            for i, image in progress_bar:
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Apply compression to image
                # For very aggressive compression, resize if image is very large
                max_dimension = 2000  # Maximum width or height in pixels
                if quality < 30 and (image.width > max_dimension or image.height > max_dimension):
                    # Calculate new dimensions maintaining aspect ratio
                    ratio = min(max_dimension / image.width, max_dimension / image.height)
                    new_width = int(image.width * ratio)
                    new_height = int(image.height * ratio)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Get image dimensions
                img_width, img_height = image.size
                
                # Determine page size
                page_width = (img_width / ocr_dpi) * 72
                page_height = (img_height / ocr_dpi) * 72
                
                # Set page size
                c.setPageSize((page_width, page_height))
                
                # Save compressed image temporarily
                temp_img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                save_kwargs = {
                    'quality': quality,
                    'optimize': True,
                }
                if quality < 50:
                    save_kwargs['progressive'] = True
                image.save(temp_img_path, 'JPEG', **save_kwargs)
                
                # Draw compressed image as background
                c.drawImage(temp_img_path, 0, 0, width=page_width, height=page_height, preserveAspectRatio=True)
                
                # Get OCR data with detailed information
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng')
                
                # Extract text with positions
                text_objects = []
                n_boxes = len(ocr_data['text'])
                
                for j in range(n_boxes):
                    text = ocr_data['text'][j].strip()
                    if text and len(text) > 0:
                        x = ocr_data['left'][j]
                        y = ocr_data['top'][j]
                        w = ocr_data['width'][j]
                        h = ocr_data['height'][j]
                        conf = int(ocr_data['conf'][j])
                        
                        if conf > 30:  # Confidence threshold
                            text_objects.append({
                                'text': text,
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h,
                                'conf': conf
                            })
                
                # Add text objects to PDF
                text_objects.sort(key=lambda t: (-t['y'], t['x']))
                
                for text_obj in text_objects:
                    # Convert image coordinates to PDF coordinates
                    pdf_x = (text_obj['x'] / ocr_dpi) * 72
                    pdf_y = page_height - ((text_obj['y'] + text_obj['h']) / ocr_dpi) * 72
                    
                    # Calculate font size to match text height
                    font_size = max(8, min(72, (text_obj['h'] / ocr_dpi) * 72 * 0.9))
                    
                    # Add text as editable text object
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont("Helvetica", font_size)
                    c.setFillAlpha(0.01)  # Almost invisible but still creates text object
                    c.drawString(pdf_x, pdf_y, text_obj['text'])
                    c.setFillAlpha(1.0)  # Reset alpha
                
                # Start new page
                c.showPage()
            
            # Save PDF
            c.save()
            
            if verbose:
                tqdm.write("✓ Compression and OCR completed successfully! PDF is now compressed and editable.")
            
            return True
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        if verbose:
            print(f"\nError during compression with OCR: {e}")
            import traceback
            traceback.print_exc()
        return False


def find_optimal_compression(
    input_path: str,
    output_path: str,
    target_size: int,
    min_quality: int = 10,
    max_quality: int = 95,
    enable_ocr: bool = False
) -> bool:
    """
    Find optimal compression settings using binary search.
    
    Args:
        input_path: Path to input PDF
        output_path: Path to output PDF
        target_size: Target size in bytes
        min_quality: Minimum JPEG quality to try
        max_quality: Maximum JPEG quality to try
    
    Returns:
        True if successful, False otherwise
    """
    original_size = get_file_size(input_path)
    
    if original_size <= target_size:
        print(f"File is already {format_size(original_size)}, smaller than target {format_size(target_size)}")
        shutil.copy2(input_path, output_path)
        return True
    
    print(f"Original size: {format_size(original_size)}")
    print(f"Target size: {format_size(target_size)}")
    print("Finding optimal compression settings...")
    print()
    
    # Try different DPI values (lower DPI = smaller file)
    # Start with higher DPI and work down for better quality
    dpi_options = [300, 250, 200, 150, 100, 75, 50]
    
    # Phase 1: Quick test to find the best DPI
    print("Phase 1: Testing DPIs to find best option...")
    best_dpi = None
    best_dpi_quality = None
    best_dpi_size = None
    
    # Test with medium quality to quickly find which DPI can reach target
    test_quality = 50
    
    # Use appropriate compression function based on OCR setting
    test_compress_func = compress_pdf_with_ocr if enable_ocr else compress_pdf
    
    # Create progress bar for DPI testing
    dpi_progress = tqdm(
        enumerate(dpi_options),
        total=len(dpi_options),
        desc="Testing DPIs",
        unit="DPI",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    
    for dpi_idx, dpi in dpi_progress:
        dpi_progress.set_postfix({"DPI": dpi, "Status": "Testing..."})
        
        temp_output = output_path + ".tmp"
        if test_compress_func(input_path, temp_output, target_size, test_quality, dpi, verbose=False):
            test_size = get_file_size(temp_output)
            
            if test_size <= target_size:
                # This DPI can reach target! Use highest DPI that works for best quality
                if best_dpi is None or dpi > best_dpi:
                    best_dpi = dpi
                    best_dpi_quality = test_quality
                    best_dpi_size = test_size
                    dpi_progress.set_postfix({"DPI": dpi, "Status": f"✓ Target reached ({format_size(test_size)})"})
                else:
                    dpi_progress.set_postfix({"DPI": dpi, "Status": "✓ Target reached (higher DPI found)"})
            else:
                dpi_progress.set_postfix({"DPI": dpi, "Status": f"✗ Too large ({format_size(test_size)})"})
            
            # Clean up
            if os.path.exists(temp_output):
                os.remove(temp_output)
        else:
            dpi_progress.set_postfix({"DPI": dpi, "Status": "✗ Failed"})
    
    dpi_progress.close()
    print()
    
    # Phase 2: Detailed binary search on the best DPI found
    best_solution = None
    
    if best_dpi is None:
        print("No DPI found that can reach target size with test quality.")
        print("Will try most aggressive settings...")
    else:
        print(f"Phase 2: Optimizing quality at DPI={best_dpi}...")
        
        # Binary search for optimal quality at the best DPI
        # We want the HIGHEST quality that still meets target size
        low, high = min_quality, max_quality
        best_quality = None
        best_size = None
        iterations = 0
        max_iterations = 15  # Prevent infinite loops
        
        # Create progress bar for quality optimization
        quality_progress = tqdm(
            total=max_iterations,
            desc="Optimizing quality",
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        while low <= high and iterations < max_iterations:
            iterations += 1
            mid_quality = (low + high) // 2
            
            quality_progress.set_postfix({"Quality": mid_quality, "Range": f"{low}-{high}"})
            quality_progress.update(1)
            
            # Create temporary output file
            temp_output = output_path + ".tmp"
            
            # Use the same compression function for testing
            if test_compress_func(input_path, temp_output, target_size, mid_quality, best_dpi, verbose=False):
                temp_size = get_file_size(temp_output)
                
                if temp_size <= target_size:
                    # This quality works! Try higher quality to see if we can get better quality
                    best_quality = mid_quality
                    best_size = temp_size
                    low = mid_quality + 1  # Try higher quality
                    quality_progress.set_postfix({"Quality": mid_quality, "Status": f"✓ OK ({format_size(temp_size)})"})
                else:
                    # File is too large, need lower quality
                    high = mid_quality - 1
                    quality_progress.set_postfix({"Quality": mid_quality, "Status": f"✗ Too large ({format_size(temp_size)})"})
            else:
                # Compression failed, try lower quality
                high = mid_quality - 1
                quality_progress.set_postfix({"Quality": mid_quality, "Status": "✗ Failed"})
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        quality_progress.close()
        
        if best_quality is not None:
            best_solution = (best_dpi, best_quality, best_size)
            print(f"  Optimal quality found: {best_quality} (size: {format_size(best_size)})")
        else:
            # Fallback to test quality if binary search didn't find better
            if best_dpi_size is not None and best_dpi_size <= target_size:
                best_solution = (best_dpi, best_dpi_quality, best_dpi_size)
                print(f"  Using test quality: {best_dpi_quality} (size: {format_size(best_dpi_size)})")
    
    print()
    
    # If we found a solution that meets the target, use it
    if best_solution is not None:
        best_dpi, best_quality, best_size = best_solution
        if best_size <= target_size:
            print(f"Applying optimal settings: DPI={best_dpi}, Quality={best_quality}")
            compress_func = compress_pdf_with_ocr if enable_ocr else compress_pdf
            if compress_func(input_path, output_path, target_size, best_quality, best_dpi, verbose=True):
                final_size = get_file_size(output_path)
                print(f"Compressed size: {format_size(final_size)}")
                reduction = ((original_size - final_size) / original_size) * 100
                print(f"Size reduction: {reduction:.1f}%")
                if enable_ocr:
                    print("✓ PDF compressed and made searchable with OCR!")
                else:
                    print("✓ Target size achieved!")
                return True
    
    # If we couldn't reach target size, try even more aggressive settings
    needs_aggressive = best_solution is None or (best_solution is not None and best_solution[2] > target_size)
    
    if needs_aggressive:
        print("Warning: Could not reach target size with standard settings.")
        print("Trying most aggressive compression (DPI=50, Quality=10)...")
        
        most_aggressive_dpi = 50
        most_aggressive_quality = 10
        
        temp_output = output_path + ".tmp"
        # Use the same compression function for aggressive compression
        if test_compress_func(input_path, temp_output, target_size, most_aggressive_quality, most_aggressive_dpi, verbose=False):
            aggressive_size = get_file_size(temp_output)
            
            # If this is better than what we had (or we had nothing), use it
            should_use_aggressive = (
                best_solution is None or 
                aggressive_size < best_solution[2] or
                (aggressive_size <= target_size and best_solution[2] > target_size)
            )
            
            if should_use_aggressive:
                # Move temp file to final output
                shutil.move(temp_output, output_path)
                final_size = aggressive_size
                
                print(f"Compressed size: {format_size(final_size)}")
                reduction = ((original_size - final_size) / original_size) * 100
                print(f"Size reduction: {reduction:.1f}%")
                if final_size <= target_size:
                    if enable_ocr:
                        print("✓ Target size achieved with aggressive compression and OCR!")
                    else:
                        print("✓ Target size achieved with aggressive compression!")
                else:
                    print(f"⚠ Warning: Even with aggressive compression, final size ({format_size(final_size)}) exceeds target ({format_size(target_size)})")
                    print(f"   This may be the smallest possible size for this PDF.")
                return True
            else:
                if os.path.exists(temp_output):
                    os.remove(temp_output)
    
    # If we still have a best solution (even if it doesn't meet target), use it
    if best_solution is not None:
        best_dpi, best_quality, best_size = best_solution
        print(f"Using best found solution: DPI={best_dpi}, Quality={best_quality}")
        compress_func = compress_pdf_with_ocr if enable_ocr else compress_pdf
        if compress_func(input_path, output_path, target_size, best_quality, best_dpi, verbose=True):
            final_size = get_file_size(output_path)
            print(f"Compressed size: {format_size(final_size)}")
            reduction = ((original_size - final_size) / original_size) * 100
            print(f"Size reduction: {reduction:.1f}%")
            if final_size <= target_size:
                if enable_ocr:
                    print("✓ Target size achieved with OCR!")
                else:
                    print("✓ Target size achieved!")
            else:
                print(f"⚠ Warning: Final size ({format_size(final_size)}) exceeds target ({format_size(target_size)})")
                print(f"   This may be the smallest possible size for this PDF.")
            return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Compress PDF files to a target size with minimal quality loss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress to 5MB
  python pdf_compressor.py input.pdf -o output.pdf -s 5MB
  
  # Compress to 500KB with OCR
  python pdf_compressor.py input.pdf -o output.pdf -s 500KB --ocr
  
  # Compress with specific quality (no target size)
  python pdf_compressor.py input.pdf -o output.pdf -q 80
  
  # Add OCR to make scanned PDF searchable
  python pdf_compressor.py scanned.pdf -o searchable.pdf --ocr
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input PDF file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output PDF file path'
    )
    
    parser.add_argument(
        '-s', '--size',
        type=str,
        default=None,
        help='Target size (e.g., 5MB, 500KB, 1000000). If not specified, uses default quality.'
    )
    
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=85,
        help='JPEG quality (1-100, higher = better quality). Only used if --size is not specified. Default: 85'
    )
    
    parser.add_argument(
        '-d', '--dpi',
        type=int,
        default=200,
        help='DPI for image conversion (lower = smaller file). Only used if --size is not specified. Default: 200'
    )
    
    parser.add_argument(
        '--ocr',
        action='store_true',
        help='Enable OCR to make scanned PDFs searchable. Requires pytesseract and PyPDF2.'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.input.lower().endswith('.pdf'):
        print("Error: Input file must be a PDF")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Parse target size if provided
    target_size = None
    if args.size:
        try:
            target_size = parse_size(args.size)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Check OCR availability if requested
    if args.ocr and not OCR_AVAILABLE:
        print("Error: OCR requested but dependencies not installed.")
        print("Install with: pip install pytesseract PyPDF2 reportlab")
        print("Also install Tesseract OCR:")
        print("  macOS: brew install tesseract")
        print("  Linux: sudo apt-get install tesseract-ocr")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)
    
    # Compress PDF
    if target_size:
        success = find_optimal_compression(args.input, args.output, target_size, enable_ocr=args.ocr)
    else:
        if args.ocr:
            print(f"Compressing with quality={args.quality}, DPI={args.dpi} and adding OCR...")
            success = compress_pdf_with_ocr(args.input, args.output, None, args.quality, args.dpi, verbose=True)
            if success:
                original_size = get_file_size(args.input)
                compressed_size = get_file_size(args.output)
                print(f"Original size: {format_size(original_size)}")
                print(f"Compressed size: {format_size(compressed_size)}")
                reduction = ((original_size - compressed_size) / original_size) * 100
                print(f"Size reduction: {reduction:.1f}%")
                print("✓ PDF compressed and made searchable with OCR!")
        else:
            print(f"Compressing with quality={args.quality}, DPI={args.dpi}...")
            success = compress_pdf(args.input, args.output, None, args.quality, args.dpi, verbose=True)
            if success:
                original_size = get_file_size(args.input)
                compressed_size = get_file_size(args.output)
                print(f"Original size: {format_size(original_size)}")
                print(f"Compressed size: {format_size(compressed_size)}")
                reduction = ((original_size - compressed_size) / original_size) * 100
                print(f"Size reduction: {reduction:.1f}%")
    
    if not success:
        print("Error: Compression failed")
        sys.exit(1)
    
    print(f"Success! Compressed PDF saved to: {args.output}")


if __name__ == '__main__':
    main()
