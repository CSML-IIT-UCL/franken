#!/bin/bash
set -euxo pipefail

# Default values
PYTHON_VERSION="3.12"
PYTORCH_VERSION="2.10.0"
ENV_FILE=".github/env-dev.yml"
ENV_NAME="test"
MICROMAMBA_DIR="${HOME}/micromamba"
RUN_PIP_INSTALL=false
PIP_CACHE_DIR="${HOME}/.cache/pip"
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
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --pip-install)
            RUN_PIP_INSTALL=true
            shift
            ;;
        --pip-cache-dir)
            PIP_CACHE_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --python VERSION        Python version (default: 3.10)"
            echo "  --pytorch VERSION       PyTorch version (default: 2.0.0)"
            echo "  --env-file FILE        Environment file (default: .github/env-dev.yml)"
            echo "  --env-name NAME        Environment name (default: test)"
            echo "  --pip-install          Run additional pip installs after environment creation"
            echo "  --pip-cache-dir DIR    Pip cache directory (default: ~/.cache/pip)"
            echo "  --help                 Show this help message"
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
echo "   Run pip install: ${RUN_PIP_INSTALL}"

# Create environment fingerprint for cache validation
generate_env_fingerprint() {
    local fingerprint="py${PYTHON_VERSION}-torch${PYTORCH_VERSION}"
    # fingerprint="${fingerprint}-$(sha256sum ${ENV_FILE} | cut -d' ' -f1 | head -c 8)"
    echo "${fingerprint}"
}

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
    export PATH="${HOME}/.local/bin:${PATH}"
    # Initialize micromamba shell hook
    eval "$(micromamba shell hook -s bash)"
}

create_environment() {
    echo "🛠 Creating environment '${ENV_NAME}'..."
    
    local create_cmd="micromamba create -y -n ${ENV_NAME} python=${PYTHON_VERSION}"
    
    if eval "${create_cmd}"; then
        # Store fingerprint for cache validation
        local fingerprint=$(generate_env_fingerprint)
        echo "${fingerprint}" > "${MICROMAMBA_DIR}/envs/${ENV_NAME}/.env_fingerprint"
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
            # Check fingerprint if exists
            if [ -f "${MICROMAMBA_DIR}/envs/${ENV_NAME}/.env_fingerprint" ]; then
                local stored_fingerprint=$(cat "${MICROMAMBA_DIR}/envs/${ENV_NAME}/.env_fingerprint")
                local current_fingerprint=$(generate_env_fingerprint)
                if [ "${stored_fingerprint}" != "${current_fingerprint}" ]; then
                    echo "⚠ Environment fingerprint mismatch, recreation needed"
                    return 1
                fi
            fi
            echo "✅ Environment verified"
            return 0
        fi
    fi
    return 1
}

gen_pip_reqs() {
    TORCH_REQ_FILE=".github/torch_reqs.txt"
    PYG_REQ_FILE=".github/pyg_reqs.txt"
    REQ_FILE=".github/reqs.txt"
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
    # Standard
    echo "torch_geometric" >> $REQ_FILE
    echo "ase" >> $REQ_FILE
    echo "numpy" >> $REQ_FILE
    echo "omegaconf" >> $REQ_FILE
    echo "e3nn" >> $REQ_FILE
    echo "requests" >> $REQ_FILE
    echo "tqdm" >> $REQ_FILE
    echo "psutil" >> $REQ_FILE
    echo "docstring_parser" >> $REQ_FILE
    echo "packaging" >> $REQ_FILE
    echo "pytest" >> $REQ_FILE
    echo "pre-commit" >> $REQ_FILE
    echo "black" >> $REQ_FILE
    echo "ruff" >> $REQ_FILE
    echo "mace-torch" >> $REQ_FILE
    echo "torch-sim-atomistic" >> $REQ_FILE
    echo "metatrain>=2026.3.1" >> $REQ_FILE
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

    python -m pip install -r $TORCH_REQ_FILE
    python -m pip install -r $REQ_FILE
    python -m pip install -r $PYG_REQ_FILE
    
    echo "pip installs completed successfully"
    return 0
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
    
    # Run pip installs if requested
    if [ "${RUN_PIP_INSTALL}" = true ]; then
        run_pip_installs
    fi
    
    # Set up environment variables for subsequent GitHub Actions steps
    if [ -n "${GITHUB_ENV:-}" ]; then
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