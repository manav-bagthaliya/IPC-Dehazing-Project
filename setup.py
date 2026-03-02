#!/usr/bin/env python3
"""
Setup script for IPC-Dehaze project
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available - will use CPU mode")
        return True
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'web_results',
        'results',
        'test_results',
        'pretrained_models'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def check_pretrained_models():
    """Check if pretrained models exist"""
    model_files = ['pretrained_models/predictor.pth', 'pretrained_models/critic.pth']
    missing_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ Found model: {model_file}")
        else:
            missing_models.append(model_file)
            print(f"⚠ Missing model: {model_file}")
    
    if missing_models:
        print("\nNote: Some pretrained models are missing.")
        print("The application will not work without these models.")
        print("Please ensure the following files exist:")
        for model in missing_models:
            print(f"  - {model}")
    
    return len(missing_models) == 0

def main():
    """Main setup function"""
    print("IPC-Dehaze Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_directories()
    
    # Check pretrained models
    models_ok = check_pretrained_models()
    
    print("\n" + "=" * 40)
    print("Setup completed!")
    
    if models_ok:
        print("\n✓ All components ready!")
        print("\nTo start the web application:")
        print("  python app.py")
        print("\nTo run inference from command line:")
        print("  python inference.py -i examples/ -o results/")
    else:
        print("\n⚠ Setup completed with warnings.")
        print("Please ensure pretrained models are available before using the application.")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
