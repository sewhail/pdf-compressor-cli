# pdfedit

CLI tool for PDF manipulation.

## Install

```bash
pip install -e .
```

For compression/OCR features:
```bash
pip install -e ".[full]"
brew install poppler tesseract  # macOS
```

## Usage

```bash
pdfedit info doc.pdf
pdfedit merge a.pdf b.pdf -o out.pdf
pdfedit split doc.pdf -o pages/
pdfedit extract doc.pdf -o out.pdf --pages 1,3,5-10
pdfedit rotate doc.pdf -o out.pdf --angle 90
pdfedit encrypt doc.pdf -o out.pdf --password secret
pdfedit decrypt doc.pdf -o out.pdf --password secret
pdfedit watermark doc.pdf -o out.pdf --text "DRAFT"
pdfedit compress doc.pdf -o out.pdf -s 5MB
pdfedit ocr scanned.pdf -o out.pdf
```

## License

MIT
