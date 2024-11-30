#!/usr/bin/env python
"""
Virtual environment setup script for UHG.
Supports:
- venv
- virtualenv
- conda
- pipenv
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

def setup_venv(env_type="venv", name="uhg-env", python_version="3.9", cuda=False):
    """Set up a virtual environment."""
    if env_type == "conda":
        # Create conda environment
        subprocess.run(["conda", "env", "create", "-f", "environment.yml"])
        
    elif env_type == "pipenv":
        # Set up pipenv
        os.environ["PIPENV_VENV_IN_PROJECT"] = "1"
        subprocess.run(["pipenv", "install", "--python", python_version])
        if cuda:
            subprocess.run(["pipenv", "install", "-e", ".[gpu]"])
        else:
            subprocess.run(["pipenv", "install", "-e", ".[cpu]"])
            
    else:  # venv or virtualenv
        # Create virtual environment
        if env_type == "venv":
            subprocess.run([sys.executable, "-m", "venv", name])
        else:
            subprocess.run(["virtualenv", "-p", f"python{python_version}", name])
        
        # Get path to pip
        if platform.system() == "Windows":
            pip_path = os.path.join(name, "Scripts", "pip")
        else:
            pip_path = os.path.join(name, "bin", "pip")
        
        # Install dependencies
        subprocess.run([pip_path, "install", "--upgrade", "pip"])
        subprocess.run([pip_path, "install", "--upgrade", "setuptools", "wheel"])
        
        if cuda:
            subprocess.run([pip_path, "install", "-e", ".[gpu]"])
        else:
            subprocess.run([pip_path, "install", "-e", ".[cpu]"])

def main():
    parser = argparse.ArgumentParser(description="Set up virtual environment for UHG")
    parser.add_argument(
        "--type",
        choices=["venv", "virtualenv", "conda", "pipenv"],
        default="venv",
        help="Type of virtual environment"
    )
    parser.add_argument(
        "--name",
        default="uhg-env",
        help="Name of virtual environment"
    )
    parser.add_argument(
        "--python-version",
        default="3.9",
        help="Python version to use"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Install with CUDA support"
    )
    args = parser.parse_args()
    
    print(f"Setting up {args.type} environment...")
    setup_venv(args.type, args.name, args.python_version, args.cuda)
    
    print("\nEnvironment setup complete!")
    print("\nTo activate the environment:")
    if args.type == "conda":
        print("  conda activate uhg")
    elif args.type == "pipenv":
        print("  pipenv shell")
    else:
        if platform.system() == "Windows":
            print(f"  {args.name}\\Scripts\\activate")
        else:
            print(f"  source {args.name}/bin/activate")

if __name__ == "__main__":
    main() 