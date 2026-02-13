from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    base = os.path.dirname(__file__)
    for pkg in ('uhg', 'UHG'):
        init_py = os.path.join(base, pkg, '__init__.py')
        if os.path.exists(init_py):
            break
    else:
        raise RuntimeError("Cannot find uhg/__init__.py or UHG/__init__.py")
    with open(init_py, 'r') as f:
        content = f.read()
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

_base = os.path.dirname(__file__)
_pkgs = find_packages()
# On Linux/git clone, package dir may be UHG; normalize to uhg for "import uhg"
_use_uhg_map = any(p.startswith("UHG") for p in _pkgs)
setup(
    name="uhg",
    version="0.3.7",
    packages=[p.replace("UHG", "uhg", 1) for p in _pkgs],
    package_dir={"uhg": "UHG"} if _use_uhg_map else {},
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "networkx>=2.5",
        "scikit-learn>=1.0.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "torch": ["torch>=1.8.0", "torch-geometric>=2.0.0"],
        "mcp": ["mcp[cli]>=1.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black",
            "flake8",
            "mypy",
            "pytest-cov",
        ],
    },
    author="UHG Library Team",
    author_email="info@uhglibrary.org",
    description="Universal Hyperbolic Geometry Library for Machine Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zbovaird/UHG-Library",
    project_urls={
        "Bug Tracker": "https://github.com/zbovaird/UHG-Library/issues",
        "Source Code": "https://github.com/zbovaird/UHG-Library",
        "Documentation": "https://github.com/zbovaird/UHG-Library/tree/main/docs"
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8"
)