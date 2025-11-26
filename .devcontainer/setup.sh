#!/bin/bash

set -e

echo "🚀 Setting up IST402 development environment..."

# Update package lists
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    nano

# Install GitHub CLI
echo "🔧 Installing GitHub CLI..."
if ! command -v gh &> /dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install gh -y
    echo "✅ GitHub CLI installed"
else
    echo "✅ GitHub CLI already installed"
fi

# Install Python packages for assignments
echo "📦 Installing Python packages for assignments..."
pip install --upgrade pip

# Install from requirements.txt
if [ -f ".devcontainer/requirements.txt" ]; then
    echo "📋 Installing packages from requirements.txt..."
    pip install -r .devcontainer/requirements.txt
    echo "✅ All packages installed from requirements.txt"
else
    echo "⚠️  requirements.txt not found, installing packages individually..."
    pip install \
        transformers \
        torch \
        sentence-transformers \
        faiss-cpu \
        langchain \
        langchain-community \
        streamlit \
        pyngrok \
        pypdf \
        pillow \
        diffusers \
        accelerate \
        soundfile \
        llama-index \
        llama-index-llms-openai \
        llama-index-embeddings-openai \
        nest-asyncio \
        openai \
        jupyter \
        jupyterlab \
        ipykernel \
        pandas \
        numpy \
        matplotlib \
        seaborn
fi

# Setup Docusaurus portfolio
echo "📚 Setting up Docusaurus portfolio..."
if [ -d "IST402/portfolio" ]; then
    cd IST402/portfolio

    # Install Node.js dependencies
    if [ -f "package.json" ]; then
        npm install
        echo "✅ Docusaurus dependencies installed"
    else
        echo "⚠️  package.json not found in IST402/portfolio"
    fi
    cd ../..
else
    echo "⚠️  IST402/portfolio directory not found"
fi

# Create Jupyter config
echo "📓 Configuring Jupyter..."
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << 'JUPYTER_CONFIG'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
JUPYTER_CONFIG

echo "✅ Development environment setup complete!"
echo ""
echo "📝 To start working:"
echo "   - Assignments: Open Jupyter notebooks in IST402/assignments/"
echo "   - Portfolio: cd IST402/portfolio && npm start"
echo "   - Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"

