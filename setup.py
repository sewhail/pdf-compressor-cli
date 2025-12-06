#!/usr/bin/env python3
"""Setup script for pdfedit CLI"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

core_requirements = [
    "pypdf>=4.0.0",
]

optional_requirements = [
    "pdf2image>=1.16.0",
    "Pillow>=10.0.0",
    "img2pdf>=0.5.0",
    "pytesseract>=0.3.10",
    "reportlab>=4.0.0",
    "tqdm>=4.66.0",
]

setup(
    name="pdfedit",
    version="2.0.0",
    author="",
    description="A CLI tool for editing, compressing, and manipulating PDF files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["pdfedit", "pdf_compressor"],
    install_requires=core_requirements,
    extras_require={
        "full": optional_requirements,
        "compress": ["pdf2image>=1.16.0", "Pillow>=10.0.0", "img2pdf>=0.5.0", "tqdm>=4.66.0"],
        "ocr": ["pytesseract>=0.3.10", "reportlab>=4.0.0", "pdf2image>=1.16.0", "Pillow>=10.0.0", "tqdm>=4.66.0"],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pdfedit=pdfedit:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
