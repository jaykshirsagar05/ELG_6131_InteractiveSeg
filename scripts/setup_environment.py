#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import torch
import shutil

def check_gpu():
    """Check if CUDA is available and print GPU information."""
    if torch.cuda.is_available():
        print("\n=== GPU Information ===")
        print(f"CUDA Available: Yes")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\nWARNING: No CUDA-capable GPU found. This will significantly impact performance.")
        return False
    return True

def create_directories():
    """Create necessary directories for the project."""
    dirs = [
        "data/raw/train",
        "data/raw/val",
        "data/raw/test",
        "data/processed/train",
        "data/processed/val",
        "data/processed/test",
        "backend/uploads",
        "backend/models",
        "logs"
    ]
    
    print("\n=== Creating Directories ===")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")

def check_nnunet():
    """Check nnU-Net installation and environment variables."""
    print("\n=== Checking nnU-Net Setup ===")
    required_vars = [
        "NNUNET_RAW_DATA_BASE",
        "NNUNET_PREPROCESSED",
        "NNUNET_RESULTS"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("WARNING: The following required environment variables are not set:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
    else:
        print("All required nnU-Net environment variables are set")

def check_dependencies():
    """Check if all required Python packages are installed."""
    print("\n=== Checking Dependencies ===")
    required_packages = [
        "torch",
        "nibabel",
        "fastapi",
        "uvicorn",
        "numpy",
        "pandas",
        "scikit-learn",
        "SimpleITK",
        "nnunet"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print("\nMissing packages. Install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def main():
    print("=== Interactive Medical Image Segmentation Setup ===")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run checks
    gpu_ok = check_gpu()
    deps_ok = check_dependencies()
    create_directories()
    check_nnunet()
    
    # Print summary
    print("\n=== Setup Summary ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"GPU Available: {'Yes' if gpu_ok else 'No'}")
    print(f"Dependencies: {'All installed' if deps_ok else 'Missing some packages'}")
    
    if not (gpu_ok and deps_ok):
        print("\nWARNING: Some checks failed. Please address the issues above.")
    else:
        print("\nSetup completed successfully!")

if __name__ == "__main__":
    main() 