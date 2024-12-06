from setuptools import setup, find_packages

setup(
    name="uhg",
    version="0.1.18",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "torch-geometric>=2.0.0",
    ],
    author="Zach Bovaird",
    author_email="zach.bovaird@example.com",
    description="Universal Hyperbolic Geometry library using pure projective operations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zachbovaird/UHG-Library",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
) 