#!/usr/bin/env python3
"""
Quick Setup Script for Solar Energy Analysis Project
Fixes common setup issues and creates missing files.

Run this first: python quick_setup.py
Then run: python scripts/run_full_analysis.py
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create all required directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "src/data",
        "src/analysis",
        "src/models",
        "src/visualization",
        "scripts",
        "notebooks", 
        "outputs/reports",
        "outputs/visualizations",
        "outputs/dashboards",
        "outputs/models",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files to make directories Python packages."""
    init_files = [
        "scripts/__init__.py",
        "src/__init__.py",
        "src/data/__init__.py", 
        "src/analysis/__init__.py",
        "src/models/__init__.py",
        "src/visualization/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

def check_required_files():
    """Check if all required Python files exist."""
    required_files = [
        "scripts/download_data.py",
        "scripts/run_full_analysis.py", 
        "scripts/create_dashboard.py",
        "src/analysis/solar_performance_analyzer.py",
        "src/models/solar_predictor.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        print("\nPlease create these files using the code I provided earlier.")
        return False
    
    return True

def create_simple_test_script():
    """Create a simple test script to verify everything works."""
    test_script = """#!/usr/bin/env python3
# Simple test to verify the project setup

import sys
from pathlib import Path

print("🧪 Testing Solar Energy Analysis Project Setup")
print("=" * 50)

# Test imports
try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError:
    print("❌ pandas not found - install with: pip install pandas")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError:
    print("❌ numpy not found - install with: pip install numpy")

try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported successfully")
except ImportError:
    print("❌ matplotlib not found - install with: pip install matplotlib")

try:
    import sklearn
    print("✅ scikit-learn imported successfully")
except ImportError:
    print("❌ scikit-learn not found - install with: pip install scikit-learn")

# Test file structure
print("\\n📁 Checking file structure...")
required_dirs = ["data", "src", "scripts", "outputs"]
for directory in required_dirs:
    if Path(directory).exists():
        print(f"✅ {directory}/ directory exists")
    else:
        print(f"❌ {directory}/ directory missing")

print("\\n🎉 Setup test complete!")
print("If all items show ✅, you're ready to run the main analysis.")
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_setup.py")

def main():
    """Main setup function."""
    print("🔧 SOLAR ENERGY ANALYSIS - QUICK SETUP")
    print("=" * 50)
    
    # Create directory structure
    print("\\n1. Creating directory structure...")
    create_directory_structure()
    
    # Create __init__.py files
    print("\\n2. Creating Python package files...")
    create_init_files()
    
    # Check for required files
    print("\\n3. Checking required Python files...")
    if not check_required_files():
        print("\\n⚠️  Some files are missing. Please create them first.")
        return False
    
    # Create test script
    print("\\n4. Creating test script...")
    create_simple_test_script()
    
    print("\\n🎉 QUICK SETUP COMPLETE!")
    print("\\nNext steps:")
    print("1. Test your setup: python test_setup.py")
    print("2. Run the analysis: python scripts/run_full_analysis.py")
    print("\\nIf you still get import errors, run:")
    print("   pip install pandas numpy matplotlib seaborn plotly scikit-learn xlsxwriter openpyxl")
    
    return True

if __name__ == "__main__":
    main()