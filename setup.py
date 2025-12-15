"""
Setup script for sixdegrees package.
This file provides fallback support for editable installs with older setuptools versions.
"""

from setuptools import setup, find_packages

# Read version from __init__.py
import os
import re

def get_version():
    """Read version from sixdegrees/__init__.py"""
    init_file = os.path.join(os.path.dirname(__file__), 'sixdegrees', '__init__.py')
    with open(init_file, 'r') as f:
        content = f.read()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return "0.1.1"

# Read README for long description
def get_long_description():
    """Read README.md for long description"""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="sixdegrees",
    version=get_version(),
    description="A package for 6-DoF seismic data processing",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Andreas Brotzer",
    author_email="rotzleffel@tutanota.com",
    url="https://github.com/andbrocode/SixDegreesOfFreedom",
    license="GPL-3.0",
    license_files=["LICENSE"],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs", "docs.*", "OLD", "OLD.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0",
        "matplotlib>=3.4.0,<4.0.0",
        "obspy>=1.3.0,<2.0.0",
        "pandas>=1.3.0,<2.0.0",
        "scikit-learn>=0.24.0,<2.0.0",
        "requests>=2.25.0,<3.0.0",
        "tqdm>=4.65.0,<5.0.0",
        "typing-extensions>=4.5.0,<5.0.0",
        "acoustics>=0.2.3",
        "pyyaml>=6.0,<7.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

