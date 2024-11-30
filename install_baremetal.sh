#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print status messages
print_status() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[!]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system dependencies based on distro
install_system_deps() {
    if command_exists apt-get; then
        # Debian/Ubuntu
        print_status "Installing system dependencies (Debian/Ubuntu)..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-dev \
            python3-pip \
            python3-venv \
            build-essential \
            git \
            cmake \
            libopenblas-dev \
            libomp-dev
    elif command_exists dnf; then
        # Fedora/RHEL
        print_status "Installing system dependencies (Fedora/RHEL)..."
        sudo dnf install -y \
            python3-devel \
            python3-pip \
            gcc \
            gcc-c++ \
            git \
            cmake \
            openblas-devel \
            blas-devel
    elif command_exists pacman; then
        # Arch Linux
        print_status "Installing system dependencies (Arch Linux)..."
        sudo pacman -Sy --noconfirm \
            python \
            python-pip \
            base-devel \
            git \
            cmake \
            openblas
    elif command_exists zypper; then
        # openSUSE
        print_status "Installing system dependencies (openSUSE)..."
        sudo zypper install -y \
            python3-devel \
            python3-pip \
            gcc \
            gcc-c++ \
            git \
            cmake \
            openblas-devel
    else
        print_error "Unsupported Linux distribution"
        exit 1
    fi
}

# Function to set up Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment if requested
    if [ "$USE_VENV" = true ]; then
        python3 -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
        print_status "Virtual environment created and activated"
    fi
    
    # Upgrade pip and install build tools
    python3 -m pip install --upgrade pip setuptools wheel
}

# Function to install CUDA if needed
setup_cuda() {
    if [ "$USE_CUDA" = true ]; then
        print_status "Setting up CUDA support..."
        
        if command_exists nvidia-smi; then
            # Get CUDA version from nvidia-smi
            CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
            
            if [ "$CUDA_VERSION" -ge 11 ]; then
                print_status "Found CUDA $CUDA_VERSION"
                # Install PyTorch with CUDA support
                python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}0
            else
                print_warning "CUDA version < 11 found. Installing CPU version..."
                USE_CUDA=false
            fi
        else
            print_warning "NVIDIA driver not found. Installing CPU version..."
            USE_CUDA=false
        fi
    fi
}

# Function to install UHG
install_uhg() {
    print_status "Installing UHG..."
    
    if [ "$USE_CUDA" = true ]; then
        python3 -m pip install -e .[gpu]
    else
        python3 -m pip install -e .[cpu]
    fi
}

# Parse command line arguments
USE_CUDA=false
USE_VENV=false
VENV_PATH="uhg-env"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            USE_CUDA=true
            shift
            ;;
        --venv)
            USE_VENV=true
            shift
            ;;
        --venv-path)
            VENV_PATH="$2"
            USE_VENV=true
            shift 2
            ;;
        *)
            print_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Main installation process
print_status "Starting UHG installation..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended. Consider using --venv option."
fi

# Install system dependencies
install_system_deps

# Set up Python
setup_python

# Set up CUDA if requested
setup_cuda

# Install UHG
install_uhg

print_status "Installation complete!"

# Print usage instructions
echo
echo "UHG has been installed successfully!"
if [ "$USE_VENV" = true ]; then
    echo "To activate the virtual environment:"
    echo "  source $VENV_PATH/bin/activate"
fi
echo
echo "To verify installation, run Python and try:"
echo "  import uhg"
echo "  manifold = uhg.LorentzManifold()"
echo
if [ "$USE_CUDA" = true ]; then
    echo "CUDA support is enabled. GPU acceleration is available."
else
    echo "CPU-only version installed. For GPU support, reinstall with --cuda flag." 