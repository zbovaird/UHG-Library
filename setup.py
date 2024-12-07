from setuptools import setup, find_packages

setup(
    name="uhg",
    use_scm_version={
        "write_to": "uhg/_version.py",
        "write_to_template": '__version__ = "{version}"',
    },
    setup_requires=['setuptools_scm'],
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "uhg": ["py.typed"],  # Include type information marker
    },
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "torch-geometric>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
    },
    author="Zach Bovaird",
    author_email="zach.bovaird@example.com",
    description="Universal Hyperbolic Geometry library using pure projective operations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zachbovaird/UHG-Library",
    project_urls={
        "Documentation": "https://uhg.readthedocs.io",
        "Source": "https://github.com/zachbovaird/UHG-Library",
        "Issues": "https://github.com/zachbovaird/UHG-Library/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    zip_safe=False,  # Required for mypy to find type hints
) 