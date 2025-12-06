#!/usr/bin/env python3
"""
pdfedit - PDF Editor CLI Tool
A comprehensive CLI tool for editing, compressing, and manipulating PDF files.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List
import tempfile
import shutil

# Core PDF operations (pypdf - pure Python)
try:
    from pypdf import PdfReader, PdfWriter, PdfMerger
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter, PdfMerger
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False

# Image-based compression (optional)
try:
    from pdf2image import convert_from_path
    from PIL import Image
    import img2pdf
    from tqdm import tqdm
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

# OCR support (optional)
try:
    import pytesseract
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
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
    
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}. Use format like '5MB', '500KB', etc.")


def parse_page_range(page_str: str, total_pages: int) -> List[int]:
    """Parse page range string like '1,3,5-10' to list of 0-indexed page numbers."""
    pages = []
    parts = page_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            start = int(start.strip())
            end = int(end.strip())
            pages.extend(range(start - 1, min(end, total_pages)))
        else:
            page = int(part)
            if 1 <= page <= total_pages:
                pages.append(page - 1)
    
    return sorted(set(pages))




def cmd_info(args):
    """Show PDF information and metadata."""
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed. Run: pip install pypdf")
        return False
    
    try:
        reader = PdfReader(args.input)
        
        print(f"File: {args.input}")
        print(f"Size: {format_size(get_file_size(args.input))}")
        print(f"Pages: {len(reader.pages)}")
        
        if reader.metadata:
            print("\nMetadata:")
            for key, value in reader.metadata.items():
                if value:
                    print(f"  {key}: {value}")
        
        if reader.is_encrypted:
            print("\n⚠ This PDF is encrypted")
        
        return True
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return False


def cmd_merge(args):
    """Merge multiple PDFs into one."""
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed. Run: pip install pypdf")
        return False
    
    try:
        merger = PdfMerger()
        
        print(f"Merging {len(args.inputs)} PDF files...")
        for pdf_path in args.inputs:
            if not os.path.exists(pdf_path):
                print(f"Error: File not found: {pdf_path}")
                return False
            print(f"  + {pdf_path}")
            merger.append(pdf_path)
        
        merger.write(args.output)
        merger.close()
        
        print(f"\n✓ Merged PDF saved to: {args.output}")
        print(f"  Size: {format_size(get_file_size(args.output))}")
        return True
    except Exception as e:
        print(f"Error merging PDFs: {e}")
        return False


def cmd_split(args):
    """Split PDF into individual pages or chunks."""
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed. Run: pip install pypdf")
        return False
    
    try:
        reader = PdfReader(args.input)
        total_pages = len(reader.pages)
        
        output_dir = args.output if args.output else os.path.dirname(args.input) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(args.input).stem
        
        print(f"Splitting {args.input} ({total_pages} pages)...")
        
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            
            output_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.pdf")
            with open(output_path, 'wb') as f:
                writer.write(f)
            print(f"  → {output_path}")
        
        print(f"\n✓ Split into {total_pages} files in: {output_dir}")
        return True
    except Exception as e:
        print(f"Error splitting PDF: {e}")
        return False


def cmd_extract(args):
    """Extract specific pages from PDF."""
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed. Run: pip install pypdf")
        return False
    
    try:
        reader = PdfReader(args.input)
        total_pages = len(reader.pages)
        
        pages = parse_page_range(args.pages, total_pages)
        
        if not pages:
            print(f"Error: No valid pages specified. PDF has {total_pages} pages.")
            return False
        
        writer = PdfWriter()
        
        print(f"Extracting pages {args.pages} from {args.input}...")
        for page_idx in pages:
            writer.add_page(reader.pages[page_idx])
            print(f"  + Page {page_idx + 1}")
        
        with open(args.output, 'wb') as f:
            writer.write(f)
        
        print(f"\n✓ Extracted {len(pages)} pages to: {args.output}")
        return True
    except Exception as e:
        print(f"Error extracting pages: {e}")
        return False


def cmd_rotate(args):
    """Rotate PDF pages."""
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed. Run: pip install pypdf")
        return False
    
    try:
        reader = PdfReader(args.input)
        writer = PdfWriter()
        
        if args.pages:
            pages_to_rotate = set(parse_page_range(args.pages, len(reader.pages)))
        else:
            pages_to_rotate = set(range(len(reader.pages)))
        
        print(f"Rotating pages by {args.angle}°...")
        
        for i, page in enumerate(reader.pages):
            if i in pages_to_rotate:
                page.rotate(args.angle)
                print(f"  ↻ Page {i + 1}")
            writer.add_page(page)
        
        with open(args.output, 'wb') as f:
            writer.write(f)
        
        print(f"\n✓ Rotated PDF saved to: {args.output}")
        return True
    except Exception as e:
        print(f"Error rotating PDF: {e}")
        return False


def cmd_encrypt(args):
    """Encrypt/password-protect a PDF."""
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed. Run: pip install pypdf")
        return False
    
    try:
        reader = PdfReader(args.input)
        writer = PdfWriter()
        
        for page in reader.pages:
            writer.add_page(page)
        
        if reader.metadata:
            writer.add_metadata(reader.metadata)
        
        writer.encrypt(args.password)
        
        with open(args.output, 'wb') as f:
            writer.write(f)
        
        print(f"✓ Encrypted PDF saved to: {args.output}")
        return True
    except Exception as e:
        print(f"Error encrypting PDF: {e}")
        return False


def cmd_decrypt(args):
    """Decrypt/remove password from a PDF."""
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed. Run: pip install pypdf")
        return False
    
    try:
        reader = PdfReader(args.input)
        
        if reader.is_encrypted:
            if not reader.decrypt(args.password):
                print("Error: Incorrect password")
                return False
        
        writer = PdfWriter()
        
        for page in reader.pages:
            writer.add_page(page)
        
        if reader.metadata:
            writer.add_metadata(reader.metadata)
        
        with open(args.output, 'wb') as f:
            writer.write(f)
        
        print(f"✓ Decrypted PDF saved to: {args.output}")
        return True
    except Exception as e:
        print(f"Error decrypting PDF: {e}")
        return False


def cmd_watermark(args):
    """Add text watermark to PDF."""
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed. Run: pip install pypdf")
        return False
    
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.pagesizes import letter
        from io import BytesIO
    except ImportError:
        print("Error: reportlab not installed. Run: pip install reportlab")
        return False
    
    try:
        reader = PdfReader(args.input)
        writer = PdfWriter()
        
        print(f"Adding watermark: '{args.text}'...")
        
        for i, page in enumerate(reader.pages):
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)
            
            packet = BytesIO()
            c = rl_canvas.Canvas(packet, pagesize=(page_width, page_height))
            
            c.setFillAlpha(args.opacity)
            c.setFillColorRGB(0.5, 0.5, 0.5)
            c.setFont("Helvetica-Bold", args.size)
            
            c.saveState()
            c.translate(page_width / 2, page_height / 2)
            c.rotate(45)
            c.drawCentredString(0, 0, args.text)
            c.restoreState()
            
            c.save()
            
            packet.seek(0)
            watermark_reader = PdfReader(packet)
            page.merge_page(watermark_reader.pages[0])
            writer.add_page(page)
        
        with open(args.output, 'wb') as f:
            writer.write(f)
        
        print(f"\n✓ Watermarked PDF saved to: {args.output}")
        return True
    except Exception as e:
        print(f"Error adding watermark: {e}")
        return False


def cmd_compress(args):
    """Compress PDF to reduce file size."""
    if not COMPRESSION_AVAILABLE:
        print("Error: Compression dependencies not installed.")
        print("Run: pip install pdf2image Pillow img2pdf tqdm")
        print("Also install poppler: brew install poppler (macOS)")
        return False
    
    from pdf_compressor import compress_pdf, find_optimal_compression, compress_pdf_with_ocr
    
    target_size = None
    if args.size:
        try:
            target_size = parse_size(args.size)
        except ValueError as e:
            print(f"Error: {e}")
            return False
    
    original_size = get_file_size(args.input)
    
    if target_size:
        success = find_optimal_compression(args.input, args.output, target_size, enable_ocr=args.ocr)
    else:
        if args.ocr:
            print(f"Compressing with quality={args.quality}, DPI={args.dpi} and adding OCR...")
            success = compress_pdf_with_ocr(args.input, args.output, None, args.quality, args.dpi, verbose=True)
        else:
            print(f"Compressing with quality={args.quality}, DPI={args.dpi}...")
            success = compress_pdf(args.input, args.output, None, args.quality, args.dpi, verbose=True)
    
    if success:
        final_size = get_file_size(args.output)
        reduction = ((original_size - final_size) / original_size) * 100
        print(f"\nOriginal: {format_size(original_size)} → Compressed: {format_size(final_size)}")
        print(f"Size reduction: {reduction:.1f}%")
        print(f"✓ Compressed PDF saved to: {args.output}")
    
    return success


def cmd_ocr(args):
    """Add OCR text layer to scanned PDF."""
    if not OCR_AVAILABLE:
        print("Error: OCR dependencies not installed.")
        print("Run: pip install pytesseract reportlab")
        print("Also install Tesseract: brew install tesseract (macOS)")
        return False
    
    from pdf_compressor import create_searchable_pdf
    
    print("Adding OCR to make PDF searchable...")
    original_size = get_file_size(args.input)
    
    success = create_searchable_pdf(args.input, args.output, dpi=args.dpi, verbose=True)
    
    if success:
        final_size = get_file_size(args.output)
        print(f"\nOriginal: {format_size(original_size)} → Output: {format_size(final_size)}")
        print(f"✓ Searchable PDF saved to: {args.output}")
    
    return success




def main():
    parser = argparse.ArgumentParser(
        prog='pdfedit',
        description='PDF Editor CLI - Edit, compress, and manipulate PDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdfedit info document.pdf
  pdfedit merge file1.pdf file2.pdf -o combined.pdf
  pdfedit split document.pdf -o output_folder/
  pdfedit extract document.pdf -o excerpt.pdf --pages 1,3,5-10
  pdfedit rotate document.pdf -o rotated.pdf --angle 90
  pdfedit compress document.pdf -o smaller.pdf -s 5MB
  pdfedit ocr scanned.pdf -o searchable.pdf
  pdfedit encrypt document.pdf -o secure.pdf --password secret
  pdfedit decrypt secure.pdf -o unlocked.pdf --password secret
  pdfedit watermark document.pdf -o marked.pdf --text "CONFIDENTIAL"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # INFO command
    info_parser = subparsers.add_parser('info', help='Show PDF information')
    info_parser.add_argument('input', help='Input PDF file')
    
    # MERGE command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple PDFs')
    merge_parser.add_argument('inputs', nargs='+', help='Input PDF files to merge')
    merge_parser.add_argument('-o', '--output', required=True, help='Output PDF file')
    
    # SPLIT command
    split_parser = subparsers.add_parser('split', help='Split PDF into pages')
    split_parser.add_argument('input', help='Input PDF file')
    split_parser.add_argument('-o', '--output', help='Output directory (default: same as input)')
    
    # EXTRACT command
    extract_parser = subparsers.add_parser('extract', help='Extract specific pages')
    extract_parser.add_argument('input', help='Input PDF file')
    extract_parser.add_argument('-o', '--output', required=True, help='Output PDF file')
    extract_parser.add_argument('--pages', required=True, help='Pages to extract (e.g., 1,3,5-10)')
    
    # ROTATE command
    rotate_parser = subparsers.add_parser('rotate', help='Rotate PDF pages')
    rotate_parser.add_argument('input', help='Input PDF file')
    rotate_parser.add_argument('-o', '--output', required=True, help='Output PDF file')
    rotate_parser.add_argument('--angle', type=int, default=90, choices=[90, 180, 270], help='Rotation angle')
    rotate_parser.add_argument('--pages', help='Pages to rotate (default: all)')
    
    # ENCRYPT command
    encrypt_parser = subparsers.add_parser('encrypt', help='Password-protect PDF')
    encrypt_parser.add_argument('input', help='Input PDF file')
    encrypt_parser.add_argument('-o', '--output', required=True, help='Output PDF file')
    encrypt_parser.add_argument('--password', required=True, help='Password to set')
    
    # DECRYPT command
    decrypt_parser = subparsers.add_parser('decrypt', help='Remove password protection')
    decrypt_parser.add_argument('input', help='Input PDF file')
    decrypt_parser.add_argument('-o', '--output', required=True, help='Output PDF file')
    decrypt_parser.add_argument('--password', required=True, help='Current password')
    
    # WATERMARK command
    watermark_parser = subparsers.add_parser('watermark', help='Add text watermark')
    watermark_parser.add_argument('input', help='Input PDF file')
    watermark_parser.add_argument('-o', '--output', required=True, help='Output PDF file')
    watermark_parser.add_argument('--text', required=True, help='Watermark text')
    watermark_parser.add_argument('--opacity', type=float, default=0.3, help='Opacity (0-1, default: 0.3)')
    watermark_parser.add_argument('--size', type=int, default=60, help='Font size (default: 60)')
    
    # COMPRESS command
    compress_parser = subparsers.add_parser('compress', help='Compress PDF')
    compress_parser.add_argument('input', help='Input PDF file')
    compress_parser.add_argument('-o', '--output', required=True, help='Output PDF file')
    compress_parser.add_argument('-s', '--size', help='Target size (e.g., 5MB, 500KB)')
    compress_parser.add_argument('-q', '--quality', type=int, default=85, help='JPEG quality (1-100)')
    compress_parser.add_argument('-d', '--dpi', type=int, default=200, help='DPI for conversion')
    compress_parser.add_argument('--ocr', action='store_true', help='Add OCR text layer')
    
    # OCR command
    ocr_parser = subparsers.add_parser('ocr', help='Add searchable text layer')
    ocr_parser.add_argument('input', help='Input PDF file')
    ocr_parser.add_argument('-o', '--output', required=True, help='Output PDF file')
    ocr_parser.add_argument('-d', '--dpi', type=int, default=300, help='DPI for OCR (default: 300)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if hasattr(args, 'input') and not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    commands = {
        'info': cmd_info,
        'merge': cmd_merge,
        'split': cmd_split,
        'extract': cmd_extract,
        'rotate': cmd_rotate,
        'encrypt': cmd_encrypt,
        'decrypt': cmd_decrypt,
        'watermark': cmd_watermark,
        'compress': cmd_compress,
        'ocr': cmd_ocr,
    }
    
    handler = commands.get(args.command)
    if handler:
        success = handler(args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
