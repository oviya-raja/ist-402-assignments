# Environment Variables Setup

## 🔒 Secure Token Management

This notebook uses environment variables to securely store your Hugging Face API token instead of hardcoding it.

## Setup Instructions

### 1. Create a `.env` file

In the same directory as this notebook, create a file named `.env` with the following content:

```env
HUGGINGFACE_HUB_TOKEN=your_actual_token_here
```

### 2. Get Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "IST402-Week3")
4. Select "Read" permissions
5. Copy the token
6. Paste it in your `.env` file

### 3. Example `.env` file

```env
# Hugging Face API Token
# Get your token from: https://huggingface.co/settings/tokens
HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## ⚠️ Important Security Notes

- ✅ **DO**: Keep your `.env` file local and never commit it to git
- ✅ **DO**: Add `.env` to `.gitignore` (already done)
- ❌ **DON'T**: Share your `.env` file or commit it to version control
- ❌ **DON'T**: Hardcode tokens in your notebook code
- ❌ **DON'T**: Share screenshots showing your token

## How It Works

The notebook uses `python-dotenv` to load environment variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env file
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
```

This ensures your token is:

- ✅ Not visible in the notebook code
- ✅ Not accidentally committed to git
- ✅ Easy to update without changing code
- ✅ Secure and follows best practices
