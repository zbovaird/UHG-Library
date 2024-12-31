from setuptools import setup, find_packages

setup(
    name="uhg",
    version="0.2.4",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "networkx>=2.5",
        "pytest>=6.0.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0"
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "mypy",
            "pytest-cov"
        ]
    },
    author="UHG Library Team",
    author_email="info@uhglibrary.org",
    description="Universal Hyperbolic Geometry Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uhg-library/uhg",
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