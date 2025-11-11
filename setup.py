"""
Setup script for CV AR Graphics package
"""

from setuptools import setup, find_packages

# Find all packages in current directory
found_packages = find_packages()

# Manually construct the package list with cv_ar_graphics prefix
packages = ['cv_ar_graphics'] + [f'cv_ar_graphics.{pkg}' for pkg in found_packages if pkg != 'cv_ar_graphics']

# Map package names to current directory structure
# This tells setuptools that cv_ar_graphics.vision maps to ./vision
package_dir = {
    'cv_ar_graphics': '.',
}
# Also map subpackages
for pkg in found_packages:
    if pkg != 'cv_ar_graphics':
        package_dir[f'cv_ar_graphics.{pkg}'] = pkg

setup(
    name="cv_ar_graphics",
    version="1.0.0",
    description="Real-Time Computer Vision AR Graphics System",
    author="AMD ATG Project",
    packages=packages,
    package_dir=package_dir,
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "pillow>=10.0.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8",
)

