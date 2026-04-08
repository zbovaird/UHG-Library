#!/usr/bin/env python
"""
Installation script for UHG in Splunk environments.
This script handles:
1. Detection of Splunk Python environment
2. Installation of dependencies
3. Configuration of UHG for Splunk
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

def get_splunk_python():
    """Get the path to Splunk's Python executable."""
    if platform.system() == "Windows":
        splunk_paths = [
            r"C:\Program Files\Splunk\bin\python.exe",
            r"C:\Program Files\SplunkUniversalForwarder\bin\python.exe"
        ]
    else:
        splunk_paths = [
            "/opt/splunk/bin/python",
            "/opt/splunkforwarder/bin/python",
            "/Applications/Splunk/bin/python"
        ]
    
    for path in splunk_paths:
        if os.path.exists(path):
            return path
    
    return sys.executable  # Default to system Python if Splunk Python not found

def install_dependencies(python_path, cuda=False):
    """Install required dependencies using Splunk's pip."""
    pip_cmd = [python_path, "-m", "pip"]
    
    # Upgrade pip
    subprocess.run([*pip_cmd, "install", "--upgrade", "pip"])
    
    # Install build dependencies
    subprocess.run([*pip_cmd, "install", "--upgrade", "setuptools", "wheel"])
    
    # Install torch and torch-geometric
    if cuda:
        subprocess.run([*pip_cmd, "install", "torch>=1.7.0+cu110", "torch-geometric>=2.0.0+cu110"])
    else:
        subprocess.run([*pip_cmd, "install", "torch>=1.7.0+cpu", "torch-geometric>=2.0.0+cpu"])
    
    # Install other dependencies
    subprocess.run([*pip_cmd, "install", "-r", "requirements.txt"])

def configure_splunk_app(app_name="uhg", app_dir=None):
    """Configure UHG as a Splunk app."""
    if app_dir is None:
        if platform.system() == "Windows":
            app_dir = r"C:\Program Files\Splunk\etc\apps"
        else:
            app_dir = "/opt/splunk/etc/apps"
    
    app_path = Path(app_dir) / app_name
    
    # Create app directory structure
    os.makedirs(app_path / "bin", exist_ok=True)
    os.makedirs(app_path / "lib", exist_ok=True)
    
    # Create app.conf
    with open(app_path / "default/app.conf", "w") as f:
        f.write("""[install]
is_configured = 1

[ui]
is_visible = 1
label = Universal Hyperbolic Geometry

[launcher]
author = Zach Bovaird
description = Universal Hyperbolic Geometry Library for PyTorch
version = 0.1.0""")
    
    # Create __init__.py in bin for Python imports
    with open(app_path / "bin/__init__.py", "w") as f:
        f.write("")
    
    # Create a wrapper script for UHG
    with open(app_path / "bin/uhg_wrapper.py", "w") as f:
        f.write("""#!/usr/bin/env python
import sys
import os

# Add UHG library path
uhg_path = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, uhg_path)

import uhg

def setup_uhg():
    \"\"\"Initialize UHG in Splunk environment.\"\"\"
    return uhg.ProjectiveUHG()  # Default implementation

if __name__ == "__main__":
    manifold = setup_uhg()
    # Add your UHG code here
""")

def main():
    parser = argparse.ArgumentParser(description="Install UHG in Splunk environment")
    parser.add_argument("--cuda", action="store_true", help="Install with CUDA support")
    parser.add_argument("--app-name", default="uhg", help="Splunk app name")
    parser.add_argument("--app-dir", help="Splunk apps directory")
    args = parser.parse_args()
    
    # Get Splunk's Python
    python_path = get_splunk_python()
    print(f"Using Python: {python_path}")
    
    # Install dependencies
    print("Installing dependencies...")
    install_dependencies(python_path, args.cuda)
    
    # Configure Splunk app
    print("Configuring Splunk app...")
    configure_splunk_app(args.app_name, args.app_dir)
    
    # Install UHG
    print("Installing UHG...")
    subprocess.run([python_path, "-m", "pip", "install", "-e", "."])
    
    print("Installation complete!")
    print("\nTo use UHG in Splunk:")
    print("1. Restart Splunk")
    print("2. Import uhg in your Splunk Python scripts:")
    print("   from uhg_wrapper import setup_uhg")
    print("   manifold = setup_uhg()")

if __name__ == "__main__":
    main() 