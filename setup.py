from setuptools import setup, find_packages

# Read version from __init__.py
with open('UHG/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"').strip("'")
            break

# Read README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="uhg",
    version=version,
    description="Universal Hyperbolic Geometry Library for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zach Bovaird",
    author_email="zach.bovaird@gmail.com",
    url="https://github.com/zachbovaird/UHG-Library",
    project_urls={
        "Documentation": "https://uhg.readthedocs.io",
        "Source": "https://github.com/zachbovaird/UHG-Library",
        "Bug Tracker": "https://github.com/zachbovaird/UHG-Library/issues",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "torch-geometric>=2.0.0",
        "networkx>=2.5",
        "tqdm>=4.50.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.7.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "twine>=3.4.0",
            "build>=0.7.0"
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "nbsphinx>=0.8.0",
            "jupyter>=1.0.0"
        ],
        "gpu": [
            "torch>=1.7.0+cu110",
            "torch-geometric>=2.0.0+cu110"
        ],
        "cpu": [
            "torch>=1.7.0+cpu",
            "torch-geometric>=2.0.0+cpu"
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Framework :: Jupyter",
        "Framework :: Pytest",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux"
    ],
    keywords=[
        "hyperbolic geometry",
        "universal hyperbolic geometry",
        "deep learning",
        "graph neural networks",
        "pytorch",
        "machine learning",
        "neural networks",
        "geometric deep learning",
        "manifold learning"
    ],
    include_package_data=True,
    zip_safe=False,
    platforms=["any"],
    entry_points={
        "console_scripts": [
            "uhg=UHG.cli:main",
        ],
    }
) 