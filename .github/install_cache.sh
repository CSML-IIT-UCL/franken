#!/bin/bash
set -euxo pipefail

# Default values
PYTHON_VERSION="3.12"
PYTORCH_VERSION="2.10.0"
ENV_NAME="test"
MICROMAMBA_DIR="${HOME}/micromamba"
PIP_CACHE_DIR="${HOME}/.cache/pip"
CONDA_CREATE_ARGS=""
export MAMBA_ROOT_PREFIX="$MICROMAMBA_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --pytorch)
            PYTORCH_VERSION="$2"
            shift 2
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --pip-cache-dir)
            PIP_CACHE_DIR="$2"
            shift 2
            ;;
        --conda-create-args)
            CONDA_CREATE_ARGS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --python VERSION            Python version (default: 3.10)"
            echo "  --pytorch VERSION           PyTorch version (default: 2.0.0)"
            echo "  --env-file FILE             Environment file (default: .github/env-dev.yml)"
            echo "  --env-name NAME             Environment name (default: test)"
            echo "  --pip-cache-dir DIR         Pip cache directory (default: ~/.cache/pip)"
            echo "  --conda-create-args ARGS    Extra arguments for the conda environment"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🔧 Setting up micromamba environment: ${ENV_NAME}"
echo "   Python: ${PYTHON_VERSION}"
echo "   PyTorch: ${PYTORCH_VERSION}"
echo "   MicroMamba environment directory: ${MICROMAMBA_DIR}"
echo "   Extra conda arguments: ${CONDA_CREATE_ARGS}"


# Function to install micromamba
install_micromamba() {
    if [ ! -f "${MICROMAMBA_DIR}/bin/micromamba" ]; then
        echo "Installing micromamba..."
        mkdir -p "${MICROMAMBA_DIR}"
        # Download and run the official install script
        if curl -Ls https://micro.mamba.pm/install.sh | bash -s -- --prefix "${MICROMAMBA_DIR}"; then
            echo "✅ Micromamba installed via official script"
        else
            echo "❌ Error:  Official install script failed"
            exit 1
        fi
    else
        echo "✅ Micromamba already installed"
    fi
    # default paths created by micromamba
    export PATH="${HOME}/.local/bin:${PATH}"
    export MAMBA_EXE="${HOME}/.local/bin/micromamba"
    # Initialize micromamba shell hook
    eval "$(micromamba shell hook -s bash)"
}

create_environment() {
    echo "🛠 Creating environment '${ENV_NAME}'..."
    echo "    Extra arguments: '${CONDA_CREATE_ARGS}'"
    
    local create_cmd="micromamba create -y -n ${ENV_NAME} python=${PYTHON_VERSION} ${CONDA_CREATE_ARGS}"
    
    if eval "${create_cmd}"; then
        echo "✅ Environment created successfully"
        return 0
    else
        echo "❌ Failed to create environment"
        return 1
    fi
}

verify_environment() {
    if [ -d "${MICROMAMBA_DIR}/envs/${ENV_NAME}" ]; then
        # Check if environment is functional
        if micromamba run -n "${ENV_NAME}" python --version &>/dev/null; then
            echo "✅ Environment verified"
            return 0
        fi
    fi
    return 1
}

gen_pip_reqs() {
    TORCH_REQ_FILE=".github/torch_reqs.txt"
    PYG_REQ_FILE=".github/pyg_reqs.txt"
    # PyTorch
    echo "--index-url https://download.pytorch.org/whl/cpu" >> $TORCH_REQ_FILE
    echo "torch==${PYTORCH_VERSION}" >> $TORCH_REQ_FILE
    # Torch Geometric
    echo "--find-links https://data.pyg.org/whl/torch-${PYTORCH_VERSION}.0+cpu.html" >> $PYG_REQ_FILE
    echo "pyg_lib" >> $PYG_REQ_FILE
    echo "torch_scatter" >> $PYG_REQ_FILE
    echo "torch_sparse" >> $PYG_REQ_FILE
    echo "torch_cluster" >> $PYG_REQ_FILE
    echo "torch_spline_conv" >> $PYG_REQ_FILE
}

run_pip_installs() {
    echo "📦 Running additional pip installs..."
    
    # Configure pip to use cache
    export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
    mkdir -p "${PIP_CACHE_DIR}"
    
    # Activate environment for pip installs
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "${ENV_NAME}"
    
    gen_pip_reqs

    python -m pip install -r $TORCH_REQ_FILE --cache-dir "$PIP_CACHE_DIR"
    # python -m pip install -r $REQ_FILE
    python -m pip install -r $PYG_REQ_FILE --cache-dir "$PIP_CACHE_DIR"
    
    echo "pip installs completed successfully"
}

# Main execution
main() {
    install_micromamba
    
    # Check if environment needs creation
    if ! verify_environment; then
        echo "🔄 Environment needs to be created/updated"
        
        # Remove existing environment if present
        if [ -d "${MICROMAMBA_DIR}/envs/${ENV_NAME}" ]; then
            echo "🗑 Removing old environment..."
            rm -rf "${MICROMAMBA_DIR}/envs/${ENV_NAME}"
        fi
        
        # Create new environment
        if ! create_environment; then
            echo "❌ Environment creation failed"
            exit 1
        fi
        
        # Set flag that environment was freshly created (for pip installs)
        ENV_FRESHLY_CREATED=true
    else
        echo "✅ Using cached environment"
        ENV_FRESHLY_CREATED=false
    fi
    
    # Run pip installs
    run_pip_installs
    
    # Set up environment variables for subsequent GitHub Actions steps
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "ENV_NAME=${ENV_NAME}" >> "${GITHUB_ENV}"
        echo "MAMBA_EXE=${MAMBA_EXE}" >> "${GITHUB_ENV}"
        echo "MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}" >> "${GITHUB_ENV}"
        echo "CONDA_PREFIX=${MICROMAMBA_DIR}/envs/${ENV_NAME}" >> "${GITHUB_ENV}"
        echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}" >> "${GITHUB_ENV}"
        
        # Also set output for the freshly created flag
        if [ -n "${GITHUB_OUTPUT:-}" ]; then
            echo "env_freshly_created=${ENV_FRESHLY_CREATED:-false}" >> "${GITHUB_OUTPUT}"
        fi
    fi
    
    echo "✅ Setup complete!"
}

# Run main function
main