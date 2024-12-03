from setuptools import setup, find_packages

setup(
    name="uhg",
    version="0.1.7",
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    package_data={
        'uhg': ['*.py'],
    },
    exclude_package_data={
        '': ['*.pdf'],
    },
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "torch-geometric>=2.0.0",
        "networkx>=2.5",
        "tqdm>=4.50.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "geoopt>=0.5.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.7.0",
            "flake8>=3.8.0",
            "mypy>=0.800"
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "nbsphinx>=0.8.0"
        ]
    }
) 