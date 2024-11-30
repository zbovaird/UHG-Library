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

# Check if running on Mac Silicon
check_architecture() {
    if [[ $(uname -m) != "arm64" ]]; then
        print_error "This script is for Apple Silicon (M1/M2) Macs only."
        exit 1
    fi
}

# Check if Conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed. Please install Miniforge for Apple Silicon first:"
        echo "Visit: https://github.com/conda-forge/miniforge#miniforge3"
        exit 1
    fi
}

# Create and activate conda environment
setup_conda_env() {
    ENV_NAME="uhg"
    if [ "$1" != "" ]; then
        ENV_NAME="$1"
    fi

    print_status "Creating conda environment: $ENV_NAME"
    
    # Remove existing environment if it exists
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    
    # Create new environment
    conda env create -f environment.mac-silicon.yml -n "$ENV_NAME"
    
    print_status "Activating conda environment"
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
}

# Install UHG
install_uhg() {
    print_status "Installing UHG..."
    
    # Install in development mode
    pip install -e .
    
    print_status "Installation complete!"
}

# Main installation process
main() {
    ENV_NAME=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENV_NAME="$2"
                shift 2
                ;;
            *)
                print_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done
    
    # Check requirements
    check_architecture
    check_conda
    
    # Set up environment
    setup_conda_env "$ENV_NAME"
    
    # Install UHG
    install_uhg
    
    # Print success message
    echo
    echo "UHG has been installed successfully!"
    echo
    echo "To activate the environment:"
    echo "  conda activate ${ENV_NAME:-uhg}"
    echo
    echo "To verify installation, run Python and try:"
    echo "  import uhg"
    echo "  manifold = uhg.LorentzManifold()"
}

main "$@" 