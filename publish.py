#!/usr/bin/env python
"""
Script to publish UHG to PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def clean_build_files():
    """Clean up build artifacts."""
    print("Cleaning build files...")
    dirs_to_remove = ['build', 'dist', 'uhg.egg-info']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            
def run_tests():
    """Run test suite."""
    print("Running tests...")
    result = subprocess.run(['pytest'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Tests failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

def build_package():
    """Build package distributions."""
    print("Building package...")
    subprocess.run([sys.executable, '-m', 'build'], check=True)

def check_distributions():
    """Check built distributions with twine."""
    print("Checking distributions...")
    result = subprocess.run(
        ['twine', 'check', 'dist/*'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Distribution check failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

def publish_to_pypi(test=False):
    """Publish to PyPI or TestPyPI."""
    repository = '--repository-url https://test.pypi.org/legacy/' if test else ''
    cmd = f'twine upload {repository} dist/*'
    
    print(f"Publishing to {'TestPyPI' if test else 'PyPI'}...")
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Upload failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Publish UHG to PyPI')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Publish to TestPyPI instead of PyPI'
    )
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip running tests'
    )
    args = parser.parse_args()

    try:
        # Clean previous builds
        clean_build_files()
        
        # Run tests unless skipped
        if not args.skip_tests:
            run_tests()
        
        # Build package
        build_package()
        
        # Check distributions
        check_distributions()
        
        # Publish
        publish_to_pypi(test=args.test)
        
        print("\nPublication successful!")
        print("\nTo install the published package:")
        if args.test:
            print("pip install --index-url https://test.pypi.org/simple/ uhg")
        else:
            print("pip install uhg")
            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 