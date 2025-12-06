# pdfedit - PDF Editor CLI

A simple command-line tool to compress PDF files to a target size with minimal quality loss.

## Features

- Compress PDFs to a specific target size
- Automatic optimization using binary search algorithm
- Minimal quality loss through intelligent compression
- **OCR support** - Make scanned PDFs searchable and editable
- Supports various size formats (MB, KB, bytes)
- Configurable quality and DPI settings

## Installation

1. Install Python 3.7 or higher

2. Clone or download this repository

3. Install the CLI utility:
```bash
pip install -e .
```

This will install all dependencies and create the `pdfedit` command that you can use from anywhere.

4. Install poppler (required for pdf2image):
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils` (Debian/Ubuntu) or `sudo yum install poppler-utils` (RHEL/CentOS)
   - **Windows**: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH

5. (Optional) Install Tesseract OCR (required for `--ocr` feature):
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr` (Debian/Ubuntu) or `sudo yum install tesseract` (RHEL/CentOS)
   - **Windows**: Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH

> **Note**: If you get a "command not found" error when running `pdfedit`, you may need to add Python's bin directory to your PATH. 
> - On macOS/Linux, add this to your `~/.zshrc` or `~/.bashrc`: `export PATH="$HOME/Library/Python/3.9/bin:$PATH"` (adjust Python version as needed)
> - Or run the command with the full path: `~/Library/Python/3.9/bin/pdfedit`

## Usage

### Compress to Target Size

Compress a PDF to a specific size (recommended):

```bash
pdfedit input.pdf -o output.pdf -s 5MB
```

```bash
pdfedit input.pdf -o output.pdf -s 500KB
```

The tool will automatically find the optimal compression settings to reach the target size.

### Compress with Custom Quality

Compress with specific quality settings (no target size):

```bash
pdfedit input.pdf -o output.pdf -q 80 -d 200
```

- `-q, --quality`: JPEG quality (1-100, higher = better quality). Default: 85
- `-d, --dpi`: DPI for image conversion (lower = smaller file). Default: 200

### OCR - Make Scanned PDFs Searchable and Editable

Add OCR to make scanned PDFs searchable and editable (can be combined with compression):

```bash
# Compress and add OCR
pdfedit scanned.pdf -o editable.pdf -s 5MB --ocr

# Just add OCR without compression
pdfedit scanned.pdf -o editable.pdf --ocr
```

The `--ocr` flag uses Tesseract OCR to extract text from scanned PDFs and creates an editable PDF. The text can be selected, copied, searched, and edited in PDF editors like Adobe Acrobat.

## Options

- `input`: Input PDF file path (required)
- `-o, --output`: Output PDF file path (required)
- `-s, --size`: Target size (e.g., `5MB`, `500KB`, `1000000`). If specified, tool will automatically optimize to reach this size.
- `-q, --quality`: JPEG quality (1-100). Only used if `--size` is not specified. Default: 85
- `-d, --dpi`: DPI for image conversion. Only used if `--size` is not specified. Default: 200
- `--ocr`: Enable OCR to make scanned PDFs searchable and editable. Requires Tesseract OCR and reportlab to be installed.

## Examples

```bash
# Compress to 5MB
pdfedit document.pdf -o compressed.pdf -s 5MB

# Compress to 500KB
pdfedit document.pdf -o compressed.pdf -s 500KB

# Compress with specific quality
pdfedit document.pdf -o compressed.pdf -q 75

# Compress with lower DPI for smaller file
pdfedit document.pdf -o compressed.pdf -q 80 -d 150

# Compress to target size with OCR
pdfedit scanned.pdf -o searchable.pdf -s 5MB --ocr

# Add OCR to scanned PDF (no compression) - makes it editable
pdfedit scanned.pdf -o editable.pdf --ocr
```

## How It Works

1. **Target Size Mode** (when `-s` is specified):
   - Uses binary search algorithm to find optimal compression settings
   - Tries different DPI values (300, 250, 200, 150, 100)
   - Adjusts JPEG quality to reach target size
   - Minimizes quality loss while meeting size requirements

2. **Quality Mode** (when `-s` is not specified):
   - Converts PDF pages to images at specified DPI
   - Compresses images with specified JPEG quality
   - Converts images back to PDF

3. **OCR Mode** (when `--ocr` is specified):
   - Converts PDF pages to images
   - Uses Tesseract OCR to extract text from each page with position information
   - Creates an editable PDF with text objects positioned correctly
   - Text is selectable, copyable, searchable, and editable in PDF editors
   - Can be combined with compression for smaller, editable PDFs

## Notes

- The tool converts PDF pages to images and back, which may slightly affect text rendering
- For best results with text-heavy PDFs, consider using the target size mode
- Lower DPI values result in smaller files but may reduce image quality
- The tool preserves the original file structure and page count
- OCR works best with clear, high-quality scanned documents
- OCR processing can be slower for large PDFs - be patient!
- For OCR, the tool uses at least 300 DPI for better text recognition accuracy
- The OCR feature creates truly editable PDFs - text can be edited in PDF editors like Adobe Acrobat, PDF Expert, etc.
- Text objects are positioned to match the original document layout

## License

MIT

