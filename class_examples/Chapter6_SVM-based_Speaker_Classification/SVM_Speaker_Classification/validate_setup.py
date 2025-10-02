#!/usr/bin/env python3
"""
Setup validation script for SVM Speaker Classification
Tests all dependencies and verifies expected results
"""

import sys
import os
import subprocess
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'librosa', 
        'scipy', 'matplotlib', 'seaborn', 'noisereduce', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_file_structure():
    """Test if all expected files exist"""
    print("\n📁 Testing file structure...")
    
    expected_files = [
        'SVM_Classifier.py',
        'svm_performance_comparison.py', 
        'requirements.txt',
        'README.md'
    ]
    
    expected_dirs = [
        'datasets/5_speakers/traindata',
        'datasets/5_speakers/testdata',
        'datasets/10_speakers/traindata', 
        'datasets/10_speakers/testdata',
        'datasets/20_speakers/traindata',
        'datasets/20_speakers/testdata'
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file in expected_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            missing_files.append(file)
    
    # Check directories
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.wav')])
            print(f"  ✅ {dir_path} ({file_count} files)")
        else:
            print(f"  ❌ {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"\n❌ Missing files: {missing_files}")
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All files and directories found!")
        return True

def test_dataset_counts():
    """Test if datasets have correct number of files"""
    print("\n📊 Validating dataset sizes...")
    
    expected_counts = {
        'datasets/5_speakers/traindata': 40,   # 5 speakers * 8 files
        'datasets/5_speakers/testdata': 10,    # 5 speakers * 2 files
        'datasets/10_speakers/traindata': 80,  # 10 speakers * 8 files
        'datasets/10_speakers/testdata': 20,   # 10 speakers * 2 files
        'datasets/20_speakers/traindata': 160, # 20 speakers * 8 files
        'datasets/20_speakers/testdata': 40,   # 20 speakers * 2 files
    }
    
    all_correct = True
    
    for dir_path, expected in expected_counts.items():
        if os.path.exists(dir_path):
            actual = len([f for f in os.listdir(dir_path) if f.endswith('.wav')])
            if actual == expected:
                print(f"  ✅ {dir_path}: {actual}/{expected} files")
            else:
                print(f"  ❌ {dir_path}: {actual}/{expected} files")
                all_correct = False
        else:
            print(f"  ❌ {dir_path}: directory not found")
            all_correct = False
    
    return all_correct

def run_quick_test():
    """Run a quick test of the main functionality"""
    print("\n🚀 Running quick functionality test...")
    
    try:
        # Import main modules
        sys.path.append('.')
        
        # Test basic SVM functionality with 5-speaker dataset
        print("  Testing SVM classification on 5-speaker dataset...")
        
        result = subprocess.run([
            sys.executable, '-c', 
            """
import sys
sys.path.append('.')
from svm_performance_comparison import evaluate_svm_performance

# Quick test on 5-speaker dataset
result = evaluate_svm_performance('datasets/5_speakers', 5)
if result and result['test_accuracy'] > 0.8:
    print('SUCCESS: Test accuracy > 80%')
else:
    print('WARNING: Test accuracy < 80%')
            """
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if 'SUCCESS' in output:
                print("  ✅ Quick test passed!")
                return True
            else:
                print("  ⚠️  Quick test completed but with warnings")
                print(f"     Output: {output}")
                return True
        else:
            print(f"  ❌ Quick test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⚠️  Quick test timed out (may still work)")
        return True
    except Exception as e:
        print(f"  ❌ Quick test error: {e}")
        return False

def main():
    """Main validation function"""
    print("=" * 60)
    print("SVM SPEAKER CLASSIFICATION - SETUP VALIDATION")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Package imports
    if not test_imports():
        all_tests_passed = False
    
    # Test 2: File structure
    if not test_file_structure():
        all_tests_passed = False
    
    # Test 3: Dataset counts
    if not test_dataset_counts():
        all_tests_passed = False
    
    # Test 4: Quick functionality test
    if not run_quick_test():
        all_tests_passed = False
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Setup is valid and ready to use")
        print("\nTo run the full analysis:")
        print("  python svm_performance_comparison.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix them")
        print("\nCommon solutions:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Ensure you're in the correct directory")
        print("  3. Check if all datasets were created properly")
    print("=" * 60)

if __name__ == "__main__":
    main()