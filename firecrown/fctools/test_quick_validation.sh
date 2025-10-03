#!/bin/bash
# Quick validation test - mimics the quick-validation job from ci.yml
# This is much faster and tests the basic workflow functionality
# EXPECTS: firecrown_developer conda environment to be already active

set -e

echo "⚡ Quick CI validation test"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "environment.yml" ]; then
    echo "❌ Error: Please run this from the firecrown repository root"
    exit 1
fi

# Check if the correct conda environment is active
if [ "$CONDA_DEFAULT_ENV" != "firecrown_developer" ]; then
    echo "❌ Error: firecrown_developer conda environment is not active"
    echo "Please run: conda activate firecrown_developer"
    exit 1
fi

echo "✓ firecrown_developer environment is active"

# Install Firecrown (quick)
echo "🔧 Installing Firecrown..."
pip install --no-deps -e .

# Quick parallel linting (sequential for clarity, but same checks as CI)
echo "🔍 Running quick linting checks..."

echo "  - Black check..."
black --check firecrown tests examples

echo "  - Flake8 check..."
flake8 firecrown examples tests

echo "✅ Quick validation completed successfully!"
echo "This matches the 2-3 minute quick-validation job from the CI workflow"