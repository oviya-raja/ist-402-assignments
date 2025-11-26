# 🔧 Troubleshooting Colab Link Issues

## Error: 404 Not Found

If you see this error when clicking the Colab link:

```
Fetch for https://api.github.com/repos/oviya-raja/ist-402-assignments/contents/... failed: 404 Not Found
```

## Common Causes & Solutions

### 1. Repository Doesn't Exist on GitHub

**Problem:** The repository `oviya-raja/ist-402-assignments` hasn't been created on GitHub yet.

**Solution:**

1. **Create the repository on GitHub:**

   - Go to https://github.com/new
   - Repository name: `ist-402-assignments`
   - Make it **Public** (required for Colab links)
   - Click "Create repository"

2. **Push your code:**

   ```bash
   git remote add origin https://github.com/oviya-raja/ist-402-assignments.git
   git branch -M main
   git push -u origin main
   ```

3. **Then use the Colab link**

---

### 2. Repository is Private

**Problem:** Colab can't access private repositories via direct links.

**Solutions:**

**Option A: Make Repository Public (Recommended)**

1. Go to your repository on GitHub
2. Settings → Scroll down to "Danger Zone"
3. Click "Change visibility" → Make public

**Option B: Use Upload Method**

- Use **Option 2** in the notebook (Upload to Colab)
- This works with private repositories

---

### 3. Wrong Branch Name

**Problem:** The link uses `main` but your repository uses `master`.

**Solution:**
Update the Colab link to use `master` instead of `main`:

```
https://colab.research.google.com/github/oviya-raja/ist-402-assignments/blob/master/IST402/assignments/W3/reference/W3__Prompt_Engineering%20w_QA%20Applications-2.ipynb
```

Or rename your branch to `main`:

```bash
git branch -M main
git push -u origin main
```

---

### 4. Wrong File Path

**Problem:** The file path in the repository doesn't match the link.

**Solution:**

1. Check the actual path in your repository
2. Update the Colab link to match the correct path
3. Or use **Option 2** (Upload) instead

---

## ✅ Quick Fix: Use Upload Method

**If the Colab link doesn't work, use this method:**

1. **Download the notebook** from your local machine
2. **Go to Google Colab**: https://colab.research.google.com/
3. **Upload**: File → Upload notebook
4. **Select the file**: `W3__Prompt_Engineering w_QA Applications-2.ipynb`
5. **Continue with setup**

This method **always works** and doesn't require GitHub!

---

## 🔍 Verify Your Repository

Check if your repository exists and is accessible:

1. **Visit**: https://github.com/oviya-raja/ist-402-assignments
2. **Check**:
   - ✅ Repository exists
   - ✅ Repository is public (or you're logged in)
   - ✅ File exists at: `IST402/assignments/W3/reference/W3__Prompt_Engineering w_QA Applications-2.ipynb`
   - ✅ Branch is `main` (or update link to `master`)

---

## 📝 Alternative: Use GitHub Gist

If you can't make the repository public, use GitHub Gist:

1. **Create a Gist**: https://gist.github.com/
2. **Upload the notebook** as a file
3. **Make it public**
4. **Use Gist link** in Colab:
   ```
   https://colab.research.google.com/gist/[YOUR_USERNAME]/[GIST_ID]/W3__Prompt_Engineering%20w_QA%20Applications-2.ipynb
   ```

---

## 🎯 Recommended Workflow

1. **First time setup:**

   - Create repository on GitHub
   - Push your code
   - Make repository public
   - Use Colab link

2. **If link doesn't work:**
   - Use **Upload method** (Option 2) - always works!
   - Or fix the repository issues above

---

**Remember:** The upload method (Option 2) is the most reliable and doesn't require GitHub at all!
