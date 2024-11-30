from setuptools import setup

# Read version from __init__.py
with open('UHG/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"').strip("'")
            break

setup(
    version=version,
    entry_points={
        "console_scripts": [
            "uhg=UHG.cli:main",
        ],
    }
) 