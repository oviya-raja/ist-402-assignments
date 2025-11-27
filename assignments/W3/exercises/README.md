# Prompt Engineering Basics - Python Script

This is a standalone Python script extracted from `W3__Prompt_Engineering_Basics.ipynb` that can be run locally.

## Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Hugging Face Token

**Option A: Using .env file (Recommended)**
```bash
# Create a .env file in this directory
echo "HUGGINGFACE_HUB_TOKEN=your_token_here" > .env
```

**Option B: Environment variable**
```bash
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

**Get your token:** https://huggingface.co/settings/tokens

### 4. Run the Script

```bash
python prompt_engineering_basics.py
```

## What This Script Does

- **Example 1**: Uses Hugging Face `pipeline` for text generation
- **Example 2**: Uses direct model loading with `AutoModelForCausalLM`
- Automatically detects CPU/GPU and optimizes settings
- Demonstrates system prompts and user messages
- Shows how to extract clean responses from model output

## Model Used

**Qwen/Qwen3-0.6B** - Latest Qwen3 series (2025)
- ~1.2GB disk space
- ~2.5GB RAM needed
- Works on systems with 7.65GB total RAM ✅

**Note**: Mistral-7B is commented out in the script (requires ~14GB RAM, too large for your system)

## Requirements

- Python 3.8+
- Hugging Face account and token
- ~2GB disk space (for model download)
- ~2.5GB available RAM
- GPU optional (CPU works fine)

## Notes

- First run will download the Qwen3-0.6B model (~1.2GB)
- CPU: 3-8 seconds per response
- GPU: 1-2 seconds per response (if available)
- The script automatically optimizes for your device
- Virtual environment (`.venv/`) is already in `.gitignore`

## If You Get "Killed" Error (Out of Memory)

### Running in DevContainer?

**Yes, devcontainers can cause memory issues!** Your container has ~8GB total RAM, but only ~1.1GB available.

**Quick Fix:**
1. **Increase Docker Desktop memory** (if using Docker Desktop):
   - Docker Desktop → Settings → Resources → Advanced
   - Increase Memory to **8GB+** (or as much as you can)
   - Apply & Restart, then rebuild devcontainer
2. **Free up memory**: Close other processes, restart VS Code/Cursor
3. **Check memory**: `free -h` (should show >2GB available)

**See**: `.devcontainer/MEMORY_SETUP.md` for detailed solutions

### Other Solutions

1. Close other applications to free RAM
2. Check available memory: `free -h`
3. Run on host machine (outside devcontainer) if you have Python 3.11+
4. Use Google Colab with GPU for larger models
