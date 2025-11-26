# How to Document Your Assignments

This guide explains how to document your work after completing each week's assignment.

---

## 📋 Step-by-Step Documentation Process

### After Completing Each Assignment:

1. **Complete the assignment first** ✅
   - Finish all required work
   - Test your implementation
   - Save your notebooks/files

2. **Open the week's folder** in Docusaurus
   - Navigate to `docs/weekX-[name]/`

3. **Fill in each taxonomy page** (in order):
   - `overview.md` - Start here
   - `remember.md` - Level 1
   - `understand.md` - Level 2
   - `apply.md` - Level 3
   - `analyze.md` - Level 4
   - `evaluate.md` - Level 5
   - `create.md` - Level 6

---

## 📝 What to Include in Each Page

### Overview Page (`overview.md`)

**Required:**
- Assignment description
- Objectives completed
- Technologies used
- **Reference to assignment files** (see below)

**Example:**
```markdown
## Assignment Files

- **Notebook:** `W3__Prompt_Engineering w_QA Applications-2.ipynb`
- **Activity:** `W3__QA_Chatbot_Activity_w_Prompt_Engineering (1).ipynb`
- **Location:** [`../../assignments/W3/`](../../../assignments/W3/) directory
```

---

### 1️⃣ Remember (`remember.md`)

**What to document:**
- Key definitions you recalled
- Technologies and tools you remembered
- Concepts and terminology

**Example:**
```markdown
## Key Definitions

- **RAG**: Retrieval-Augmented Generation - combines retrieval with LLM generation
- **FAISS**: Facebook AI Similarity Search - efficient vector similarity search library
- **Embeddings**: Vector representations of text that capture semantic meaning

## Technologies Recalled

- **Mistral-7B**: Open-source language model for Q&A generation
- **sentence-transformers**: Library for generating text embeddings
- **Hugging Face Transformers**: Platform for accessing pre-trained models
```

---

### 2️⃣ Understand (`understand.md`)

**What to document:**
- How systems work (in your own words)
- Why approaches are effective
- Relationships between concepts

**Example:**
```markdown
## How RAG Works

I understand that RAG works in three stages:

1. **Retrieve**: Convert user query to embeddings, search vector database
2. **Augment**: Combine retrieved context with user query
3. **Generate**: LLM uses augmented context to produce accurate responses

## Why Embeddings Matter

I understand that vector embeddings capture semantic meaning, allowing 
the system to find relevant information even when exact keywords don't match.
```

---

### 3️⃣ Apply (`apply.md`)

**What to document:**
- Implementation steps
- Code snippets (with explanations)
- Screenshots of working system

**Example:**
```markdown
## Implementation Steps

### Step 1: Generate Q&A Database

I used Mistral-7B to generate Q&A pairs for my business context:

```python
from transformers import pipeline

qa_generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
# Generated 15 Q&A pairs for Tech Startup context
```

**Reference:** See `W3__Prompt_Engineering w_QA Applications-2.ipynb` (cells 3-5)

### Step 2: Build FAISS Index

I implemented vector storage using FAISS:

```python
import faiss
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = encoder.encode(questions)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
```

**Reference:** See `W3__QA_Chatbot_Activity_w_Prompt_Engineering (1).ipynb` (cells 8-12)
```

---

### 4️⃣ Analyze (`analyze.md`)

**What to document:**
- Comparisons made
- Performance metrics
- Component breakdowns

**Example:**
```markdown
## Model Comparison

I analyzed 6 different QA models:

| Model | Accuracy | Response Time | Quality Score |
|-------|----------|---------------|---------------|
| Mistral-7B | 85% | 1.2s | High |
| FLAN-T5 | 78% | 0.8s | Medium |
| GPT-2 | 72% | 0.6s | Medium |

**Reference:** Analysis results in `W3__QA_Chatbot_Activity_w_Prompt_Engineering (1).ipynb` (cells 20-25)

## Similarity Score Analysis

I analyzed that answerable questions had similarity scores > 0.85, 
while unanswerable questions scored < 0.60.
```

---

### 5️⃣ Evaluate (`evaluate.md`)

**What to document:**
- Performance judgments
- Critiques of your approach
- Justifications for decisions

**Example:**
```markdown
## Model Ranking

1. **Mistral-7B** - Best overall (85% accuracy, good response quality)
2. **FLAN-T5** - Fast but lower accuracy
3. **GPT-2** - Fastest but lowest accuracy

## Performance Assessment

I evaluated that the system achieved 85% accuracy on test questions. 
Limitations include handling ambiguous queries and edge cases.

**Reference:** Evaluation metrics in `W3__QA_Chatbot_Activity_w_Prompt_Engineering (1).ipynb` (cell 30)
```

---

### 6️⃣ Create (`create.md`)

**What to document:**
- Final deliverables
- Original contributions
- Portfolio-ready work

**Example:**
```markdown
## Final Deliverables

### 1. Q&A System
- Custom business context: Tech Startup - AI Consultant
- 15 Q&A pairs generated using Mistral-7B
- FAISS vector database implementation

### 2. Model Comparison Framework
- Automated comparison of 6 QA models
- Performance metrics dashboard
- Ranking system

**Reference:** Complete implementation in:
- `W3__Prompt_Engineering w_QA Applications-2.ipynb`
- `W3__QA_Chatbot_Activity_w_Prompt_Engineering (1).ipynb`

## Original Contributions

- Custom system prompt design for business context
- Optimized FAISS index configuration
- End-to-end RAG pipeline implementation
```

---

## 🔗 How to Reference Assignment Files

### Option 1: Relative Path (Recommended)

If your assignment files are in the parent directory:

```markdown
**Reference:** See [`../../assignments/W3/W3__Prompt_Engineering w_QA Applications-2.ipynb`](../../../assignments/W3/W3__Prompt_Engineering%20w_QA%20Applications-2.ipynb)

**Note:** From portfolio docs, use `../../assignments/` to reference assignment files. The portfolio is at `IST402/portfolio/` and assignments are at `IST402/assignments/`.
```

### Option 2: File Name with Location

If files are in a known location:

```markdown
**Assignment Files:**
- `W3__Prompt_Engineering w_QA Applications-2.ipynb`
- `W3__QA_Chatbot_Activity_w_Prompt_Engineering (1).ipynb`

**Location:** [`IST402/assignments/W3/`](../../../assignments/W3/) directory (relative to project root)
```

### Option 3: Link to GitHub (If Published)

If your code is on GitHub:

```markdown
**Reference:** [View Notebook on GitHub](https://github.com/oviya-raja/ist-402-assignments/blob/main/IST402/assignments/W3/W3__Prompt_Engineering%20w_QA%20Applications-2.ipynb)
```

### Option 4: Google Colab Link

If you used Google Colab:

```markdown
**Reference:** [Open in Google Colab](https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID)
```

---

## 📸 Adding Screenshots and Images

### Where to Add Images:

1. Create `static/img/weekX/` folder in the portfolio root
2. Add your screenshots there (e.g., `static/img/week3/rag-results.png`)
3. Reference them in markdown:

```markdown
![RAG System Results](/img/week3/rag-results.png)

**Screenshot:** Results from my RAG system implementation
```

**Note:** Use absolute paths starting with `/img/` for images in the `static/` folder.

---

## ✅ Documentation Checklist

After completing each assignment, ensure you have:

- [ ] **Overview** - Assignment description and file references
- [ ] **Remember** - Key concepts and definitions
- [ ] **Understand** - How systems work (in your own words)
- [ ] **Apply** - Implementation steps with code snippets
- [ ] **Analyze** - Comparisons and performance metrics
- [ ] **Evaluate** - Judgments and assessments
- [ ] **Create** - Final deliverables and original work
- [ ] **References** - Links to assignment files/notebooks
- [ ] **Screenshots** - Visual proof of working systems (optional but recommended)

---

## 💡 Tips for Effective Documentation

1. **Be Specific** - Include actual numbers, metrics, and results
2. **Use Code Snippets** - Show your implementation, not just describe it
3. **Add Screenshots** - Visual proof is powerful
4. **Reference Everything** - Link to your assignment files
5. **Be Honest** - Document challenges and limitations too
6. **Write in First Person** - "I built...", "I understood...", "I analyzed..."
7. **Keep It Organized** - Use headers, lists, and tables
8. **Update as You Go** - Don't wait until the end, document while it's fresh

---

## 📚 Example: Week 3 Documentation Structure

```
week3-prompt-engineering/
├── overview.md          # Assignment intro + file references
├── remember.md         # Level 1: Definitions & concepts
├── understand.md       # Level 2: How RAG works
├── apply.md           # Level 3: Implementation with code
├── analyze.md         # Level 4: Model comparisons
├── evaluate.md        # Level 5: Performance assessment
└── create.md          # Level 6: Final deliverables
```

---

## 🚀 Quick Start Template

When you're ready to document, copy this template:

```markdown
# Week X: [Taxonomy Level]

## What I [Action]:

[Your content here]

## Reference

**Assignment Files:**
- `[filename].ipynb` - [Description]
- `[filename].pdf` - [Description]

**Location:** [`IST402/assignments/WX/`](../../../assignments/) directory (relative to project root)
```

---

**Ready to document?** Start with the Overview page, then work through each taxonomy level! 🎯

