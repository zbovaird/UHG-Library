from setuptools import setup, find_packages

setup(
    name="uhg",
    version="0.1.0",
    description="Universal Hyperbolic Geometry Library for PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Zach Bovaird",
    author_email="zach.bovaird@gmail.com",
    url="https://github.com/zachbovaird/UHG-Library",
    packages=find_packages(),
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
            "mypy>=0.800"
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
        "Operating System :: OS Independent"
    ],
    keywords=[
        "hyperbolic geometry",
        "universal hyperbolic geometry",
        "deep learning",
        "graph neural networks",
        "pytorch",
        "machine learning"
    ]
) 