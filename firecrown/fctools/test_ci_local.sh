#!/bin/bash
# Local test script for macOS Python 3.13 CI workflow
# This mimics the firecrown-miniforge job from ci.yml
# EXPECTS: firecrown_developer conda environment to be already active

set -e  # Exit on any error

echo "🧪 Testing CI workflow locally (macOS Python 3.13 equivalent)"
echo "============================================================="

# Check if we're in the right directory
if [ ! -f "environment.yml" ]; then
    echo "❌ Error: Please run this from the firecrown repository root"
    exit 1
fi

# Check if the correct conda environment is active
if [ "$CONDA_DEFAULT_ENV" != "firecrown_developer" ]; then
    echo "❌ Error: firecrown_developer conda environment is not active"
    echo "Please run: conda activate firecrown_developer"
    echo "If the environment doesn't exist, create it with: conda env create -f environment.yml"
    exit 1
fi

echo "✓ firecrown_developer environment is active"

# Step 1: Setup Firecrown
echo "📦 Step 1: Setting up Firecrown..."
export FIRECROWN_DIR=${PWD}

# Step 2: Install Firecrown
echo "🔧 Step 2: Installing Firecrown..."
pip install --no-deps -e .

# Step 3: Setup dependencies (parallel in CI, sequential here)
echo "🛠️  Step 3: Setting up dependencies..."
# Check if dependencies are already installed
if [ ! -d "${CONDA_PREFIX}/cosmosis-standard-library" ]; then
    echo "Installing CosmoSIS Standard Library..."
    source ${CONDA_PREFIX}/bin/cosmosis-configure
    pushd ${CONDA_PREFIX}
    cosmosis-build-standard-library
    export CSL_DIR=${PWD}/cosmosis-standard-library
    popd
fi

if ! python -c "import cobaya" 2>/dev/null; then
    echo "Installing Cobaya..."
    python -m pip install cobaya
fi

# Step 4: Code quality checks (parallel in CI, sequential here for clarity)
echo "🔍 Step 4: Running code quality checks..."
echo "  - Running black check..."
black --check firecrown tests examples

echo "  - Running flake8..."
flake8 firecrown examples tests

echo "  - Running mypy..."
mypy -p firecrown -p examples -p tests

echo "  - Running pylint on firecrown..."
pylint firecrown

echo "  - Running pylint on tests..."
pylint --rcfile tests/pylintrc tests

echo "  - Running pylint on examples..."
pylint --rcfile examples/pylintrc examples

# Step 5: Test isitgr (macOS specific, but adapted for local)
echo "🧩 Step 5: Testing isitgr import..."
python -c "import isitgr; print('✓ isitgr imported successfully')"

# Step 6: Run tests (this would be the non-coverage version for non-coverage jobs)
echo "🧪 Step 6: Running unit tests..."
echo "Note: Running with --runslow since this simulates a non-coverage matrix job"
python -m pytest -vv --runslow

# Step 7: Run example tests
echo "📋 Step 7: Running example tests..."
python -m pytest -vv -s --example -m example tests/example

echo ""
echo "✅ Local CI test completed successfully!"
echo "This simulates the macOS Python 3.13 matrix job (non-coverage version)"