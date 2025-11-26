# Google Colab Setup Guide

## 📋 Quick Start Steps

### Step 1: Open Notebook in Google Colab

**Method 1: Direct Colab Link (Recommended - If notebook is on GitHub)**

- ✅ Click the **"Open in Colab"** badge in the notebook header
- Direct link: https://colab.research.google.com/github/oviya-raja/ist-402-assignments/blob/main/IST402/assignments/W3/reference/W3__Prompt_Engineering%20w_QA%20Applications-2.ipynb
- This opens the notebook directly in Colab with one click!

**Method 2: Upload to Colab**

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Select `W3__Prompt_Engineering w_QA Applications-2.ipynb`
4. Or drag and drop the file

**Method 3: Open from GitHub**

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Open notebook**
3. Go to **GitHub** tab
4. Enter your repository URL: `https://github.com/oviya-raja/ist-402-assignments`
5. Navigate to: `IST402/assignments/W3/reference/`
6. Select `W3__Prompt_Engineering w_QA Applications-2.ipynb`

---

### Step 2: Enable GPU Runtime

**This is CRITICAL for performance!**

1. Click **Runtime** → **Change runtime type**
2. In the popup:
   - **Hardware accelerator**: Select **GPU** (T4 is free tier)
   - **Runtime shape**: Standard (free) or High-RAM (if needed)
3. Click **Save**
4. **Restart runtime**: Runtime → Restart runtime

**Verify GPU is enabled:**

```python
# Run this in a new cell to check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

### Step 3: Set Up Environment Variables

**Option A: Using .env file (Recommended)**

1. **Create a new cell** and run:

```python
# Create .env file in Colab
from google.colab import files
import os

# Create .env content
env_content = "HUGGINGFACE_HUB_TOKEN=your_token_here"

# Write to file
with open('.env', 'w') as f:
    f.write(env_content)

print("✅ .env file created")
print("⚠️  Don't forget to replace 'your_token_here' with your actual token!")
```

2. **Edit the .env file:**
   - Click on the file in the left sidebar
   - Edit the token value
   - Get your token from: https://huggingface.co/settings/tokens

**Option B: Direct environment variable (Quick)**

```python
# Set token directly (less secure but faster)
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_actual_token_here"
print("✅ Token set")
```

---

### Step 4: Run the Notebook Cells

**Execute cells in order:**

1. **Cell 1**: Install core packages

   - Should complete in 1-2 minutes
   - ✅ All packages installed

2. **Cell 2**: Install FAISS

   - Will detect GPU and install `faiss-gpu` automatically
   - ✅ FAISS installed

3. **Cell 3**: Load environment variables

   - Loads token from .env file
   - ✅ Token loaded

4. **Cell 4**: Import libraries

   - Imports all required libraries
   - ✅ Libraries imported

5. **Cell 5**: Device configuration

   - Should detect GPU if runtime is set correctly
   - ✅ GPU detected and configured

6. **Cell 6+**: Continue with model loading and exercises

---

## 🔧 Troubleshooting

### Colab Link Returns 404 Error

**Problem:** Clicking the Colab badge shows "404 Not Found" error.

**Common Causes:**

1. Repository doesn't exist on GitHub yet
2. Repository is private (Colab can't access private repos via direct links)
3. Wrong branch name (`main` vs `master`)
4. File path doesn't match

**Quick Fix:**

- Use **Method 2 (Upload to Colab)** - this always works!
- Or see `TROUBLESHOOTING_COLAB.md` for detailed solutions

---

## 🔧 Other Troubleshooting

### GPU Not Detected

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions:**

1. ✅ Check Runtime → Change runtime type → GPU is selected
2. ✅ Restart runtime: Runtime → Restart runtime
3. ✅ Verify in a new cell:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
4. ✅ If still not working, try:
   ```python
   # Reinstall PyTorch with CUDA
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### FAISS Installation Fails

**Problem**: `faiss-gpu` installation fails

**Solutions:**

1. ✅ Try installing `faiss-cpu` instead (works on GPU too, just slower):
   ```python
   !pip install faiss-cpu
   ```
2. ✅ Restart runtime after installation
3. ✅ Re-run the import cell

### Token Not Found

**Problem**: `HUGGINGFACE_HUB_TOKEN not found`

**Solutions:**

1. ✅ Make sure .env file exists in the same directory
2. ✅ Check .env file has correct format: `HUGGINGFACE_HUB_TOKEN=your_token`
3. ✅ No spaces around the `=` sign
4. ✅ Re-run Cell 3 after fixing .env file

### Out of Memory

**Problem**: CUDA out of memory error

**Solutions:**

1. ✅ Restart runtime: Runtime → Restart runtime
2. ✅ Use smaller model or reduce `max_new_tokens`
3. ✅ Clear GPU memory:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
4. ✅ Use High-RAM runtime: Runtime → Change runtime type → High-RAM

---

## 📝 Colab-Specific Tips

### 1. **Session Management**

- Colab sessions timeout after ~90 minutes of inactivity
- Save your work frequently: File → Save
- Download important outputs before session ends

### 2. **File Management**

- Files uploaded to Colab are temporary (deleted when session ends)
- Use Google Drive for persistent storage:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

### 3. **Performance Optimization**

- ✅ Always use GPU runtime for AI models
- ✅ Restart runtime if it becomes slow
- ✅ Clear variables if running out of memory:
  ```python
  del large_variable
  import gc
  gc.collect()
  ```

### 4. **Package Installation**

- Use `!pip install` or `%pip install` in cells
- Installations persist during the session
- Reinstall if you restart runtime

---

## 🚀 Quick Setup Checklist

- [ ] Notebook uploaded to Colab
- [ ] GPU runtime enabled (Runtime → Change runtime type → GPU)
- [ ] Runtime restarted
- [ ] Hugging Face token obtained from https://huggingface.co/settings/tokens
- [ ] .env file created with token (or token set directly)
- [ ] Cell 1 executed (packages installed)
- [ ] Cell 2 executed (FAISS installed)
- [ ] Cell 3 executed (token loaded)
- [ ] Cell 4 executed (libraries imported)
- [ ] Cell 5 executed (GPU detected ✅)

---

## 📚 Additional Resources

- **Google Colab Docs**: https://colab.research.google.com/notebooks/intro.ipynb
- **Hugging Face Tokens**: https://huggingface.co/settings/tokens
- **PyTorch CUDA**: https://pytorch.org/get-started/locally/
- **FAISS Documentation**: https://github.com/facebookresearch/faiss

---

## 💡 Pro Tips

1. **Save a copy to Google Drive** for easy access later
2. **Use Colab Pro** ($10/month) for:
   - Faster GPUs (V100, A100)
   - Longer session times
   - More memory
3. **Monitor GPU usage**:
   ```python
   !nvidia-smi
   ```
4. **Download outputs** before session ends
5. **Share notebook** with others: File → Share

---

## 🔗 Quick Links

### Create Your Colab Link

If your notebook is on GitHub, create a direct Colab link:

**Format:**

```
https://colab.research.google.com/github/[USERNAME]/[REPO]/blob/[BRANCH]/[PATH]/[NOTEBOOK].ipynb
```

**Your Link:**

```
https://colab.research.google.com/github/oviya-raja/ist-402-assignments/blob/main/IST402/assignments/W3/reference/W3__Prompt_Engineering%20w_QA%20Applications-2.ipynb
```

**Steps:**

1. Replace `[USERNAME]` with your GitHub username
2. Replace `[REPO]` with your repository name
3. Replace `[BRANCH]` with your branch (usually `main` or `master`)
4. Replace `[PATH]` with the file path (URL-encoded)
5. Replace `[NOTEBOOK]` with the notebook filename (URL-encoded)

**URL Encoding:**

- Spaces become `%20`
- Example: `W3__Prompt_Engineering w_QA Applications-2.ipynb` → `W3__Prompt_Engineering%20w_QA%20Applications-2.ipynb`

### Colab Badge Status

✅ **The notebook header (Cell 0) already includes a Colab badge with your repository link!**

**Your Colab Link:**

```
https://colab.research.google.com/github/oviya-raja/ist-402-assignments/blob/main/IST402/assignments/W3/reference/W3__Prompt_Engineering%20w_QA%20Applications-2.ipynb
```

**Test it:** Click the "Open in Colab" badge in the notebook header - it should open directly in Google Colab!

### Quick Test:

1. Copy your Colab link
2. Paste it in a browser
3. It should open the notebook directly in Google Colab! ✅

---

**Happy coding! 🎉**
