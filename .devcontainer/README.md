# DevContainer Setup for IST402

This devcontainer provides a complete development environment for:
- **Assignments** - Python, Jupyter notebooks, AI/ML libraries
- **Portfolio** - Docusaurus website

## 🚀 Quick Start

1. **Open in VS Code with Dev Containers:**
   - Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   - Open the command palette (Cmd/Ctrl + Shift + P)
   - Select "Dev Containers: Reopen in Container"

2. **Or use GitHub Codespaces:**
   - Click "Code" → "Codespaces" → "Create codespace"

## 📦 What's Included

### Python Environment
- Python 3.11
- All required packages for assignments:
  - transformers, torch, sentence-transformers
  - faiss-cpu, langchain
  - streamlit, pyngrok
  - llama-index, openai
  - jupyter, ipykernel

### Node.js Environment
- Node.js 20
- npm for Docusaurus

### VS Code Extensions
- Python
- Jupyter
- Prettier
- Pylance

## 🔧 Usage

### Running Assignments

**Jupyter Lab:**
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```
Access at: http://localhost:8888

**Jupyter Notebook:**
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

### Running Portfolio

```bash
cd IST402/portfolio
npm start
```
Access at: http://localhost:3000

## 📁 Project Structure

```
/workspace/
├── IST402/
│   ├── assignments/    # Your assignment notebooks
│   └── portfolio/       # Docusaurus portfolio
└── README.md
```

## 🔐 Environment Variables

Set these in `.devcontainer/devcontainer.json` or use VS Code secrets:

- `HUGGINGFACE_HUB_TOKEN` - For Hugging Face models
- `OPENAI_API_KEY` - For OpenAI API (Weeks 9-10)

## 🛠️ Troubleshooting

**Port conflicts:**
- Change ports in `devcontainer.json` if 3000 or 8888 are in use

**Missing packages:**
- Run: `pip install <package>` or `npm install <package>`

**Reset environment:**
- Rebuild container: "Dev Containers: Rebuild Container"

## 📚 Resources

- [Dev Containers Docs](https://containers.dev/)
- [VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)

