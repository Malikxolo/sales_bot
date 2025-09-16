#!/usr/bin/env python3
"""
Dependency Fix Script for Brain-Heart Deep Research System
"""
import subprocess
import sys
import os

def install_package(package):
    """Install package using current Python's pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_import(module_name):
    """Check if module can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    print("🔧 Brain-Heart System Dependency Fixer")
    print("=" * 50)
    
    # Required packages
    packages = [
        'aiohttp',
        'python-dotenv',
        'streamlit', 
        'requests',
        'pandas',
        'numpy'
    ]
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check current status
    print("\n📋 Checking current dependencies:")
    missing = []
    for package in packages:
        if check_import(package.replace('-', '_')):
            print(f"✅ {package}")
        else:
            print(f"❌ {package}")
            missing.append(package)
    
    if not missing:
        print("\n🎉 All dependencies are installed!")
        return
    
    print(f"\n🔧 Installing {len(missing)} missing packages...")
    
    # Try to install missing packages
    for package in missing:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed")
        else:
            print(f"❌ Failed to install {package}")
    
    # Verify installation
    print("\n🧪 Verifying installation:")
    all_good = True
    for package in packages:
        if check_import(package.replace('-', '_')):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} still missing")
            all_good = False
    
    if all_good:
        print("\n🎉 All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Create .env file: cp .env.example .env")
        print("2. Add API key to .env")
        print("3. Run: python test_system.py")
        print("4. Launch: streamlit run app.py")
    else:
        print("\n⚠️  Some packages still missing. Try:")
        print("conda install -c conda-forge aiohttp python-dotenv streamlit requests pandas numpy")

if __name__ == "__main__":
    main()