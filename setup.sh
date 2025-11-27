#!/bin/bash

# IST402 - Virtual Environment Setup Script
# This script creates a virtual environment and installs all required packages

set -e  # Exit on any error

echo "🚀 IST402 - Setting up Python virtual environment..."
echo ""

# Check if we're in the root directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found!"
    echo "   Please run this script from the repository root directory:"
    echo "   ./setup.sh"
    exit 1
fi

# Step 1: Create virtual environment
echo "📦 Step 1: Creating virtual environment (.venv)..."
python3 -m venv .venv
echo "✅ Virtual environment created successfully!"
echo ""

# Step 2: Activate virtual environment
echo "🔌 Step 2: Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated!"
echo ""

# Step 3: Upgrade pip
echo "⬆️  Step 3: Upgrading pip to latest version..."
pip install --upgrade pip
echo "✅ pip upgraded successfully!"
echo ""

# Step 4: Install requirements
echo "📚 Step 4: Installing packages from requirements.txt..."
echo "   This may take a few minutes..."
pip install -r requirements.txt
echo "✅ All packages installed successfully!"
echo ""

# Step 5: Verify installation
echo "🔍 Step 5: Verifying installation..."
python -c "import torch; import transformers; import faiss; import streamlit; print('✅ Core packages verified!')" 2>/dev/null || echo "⚠️  Some packages may need manual verification"
echo ""

# Success message
echo "🎉 Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Activate the environment: source .venv/bin/activate"
echo "   2. Start Jupyter: jupyter lab"
echo "   3. Work on assignments in assignments/"
echo ""
echo "💡 To deactivate the environment later, run: deactivate"
echo ""

