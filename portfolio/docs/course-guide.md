# IST402 - Course Guide

Complete guide for **IST402: AI Agents, Retrieval-Augmented Generation (RAG), and Modern LLM Applications**.

> **Note:** This is a reference guide for the course. For portfolio documentation, see [Introduction](./introduction) and individual week pages.

---

## 🚀 Quick Start (30 minutes)

### 1. Set Up Accounts
- **Hugging Face**: https://huggingface.co/join → Get token at https://huggingface.co/settings/tokens
- **Google Colab**: https://colab.research.google.com (free GPU access)
- **OpenAI**: https://platform.openai.com (for Weeks 9-10, requires payment)
- **ngrok**: https://dashboard.ngrok.com (for Week 8, free tier works)

### 2. Install Packages
Open Google Colab and run:
```python
!pip install transformers torch sentence-transformers faiss-cpu langchain langchain-community
!pip install streamlit pyngrok pypdf pillow diffusers accelerate soundfile
!pip install llama-index llama-index-llms-openai llama-index-embeddings-openai nest-asyncio openai
```

### 3. Test Setup
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
from transformers import pipeline
print("✅ Setup complete!")
```

---

## 📅 Recommended Learning Path

**Note:** While weeks are numbered 3, 6, 7, 8, 9, 10, 11, here's the optimal order to tackle them:

1. **[Week 3 - Prompt Engineering & QA](./week3-prompt-engineering/overview)** (8-12h) ← **START HERE**
2. **[Week 8 - Multimodal AI](./week8-multimodal/overview)** (10-15h)
3. **[Week 9 - LlamaIndex Basics](./week9-llamaindex/overview)** (12-16h)
4. **[Week 10 - Advanced RAG](./week10-advanced-rag/overview)** (12-16h)
5. **[Week 7 - Group Project](./week7-group-assignment/overview)** (15-20h)
6. **[Week 6 - n8n Agents (Optional)](./week6-ai-agents-n8n/overview)** (4-6h)

---

## 📖 Week-by-Week Guide

### **[Week 3: Prompt Engineering & QA](./week3-prompt-engineering/overview)**
**Files:** `W3__Prompt_Engineering w_QA Applications-2.ipynb`, `W3__QA_Chatbot_Activity_w_Prompt_Engineering (1).ipynb`

**What I Have Learned:**
- Designed system prompts for LLMs
- Built Q&A databases using Mistral-7B
- Implemented FAISS vector database
- Created RAG (Retrieval-Augmented Generation) systems
- Compared multiple QA models

**Steps:**
1. Choose a business context (e.g., "Tech Startup - AI Consultant")
2. Create system prompt defining AI role
3. Generate 10-15 Q&A pairs using Mistral
4. Build FAISS index for similarity search
5. Test with answerable/unanswerable questions
6. Compare 4+ QA models and rank them

**Key Concepts:** System prompts, Vector embeddings, FAISS, RAG pipeline

---

### **[Week 6: AI Agents with n8n](./week6-ai-agents-n8n/overview)** (Optional)
**Files:** `W6-AI Agents-n8n to-do-task.pptx`

**What I Have Learned:**
- Workflow automation with n8n
- Task management systems
- Agent orchestration

---

### **[Week 7: Group Assignment](./week7-group-assignment/overview)**
**Files:** `W7GroupAssignmentAgentsDevwithOpenAI.pdf`

**What I Have Learned:**
- Built production-ready AI agents
- Integrated OpenAI API
- Collaborated on team projects

---

### **[Week 8: Multimodal AI Applications](./week8-multimodal/overview)**
**Files:** `W8_image_caption.ipynb`, `W8_pdf_Q&A.ipynb`, `W8_Speech_to_Image.ipynb`

**What I Have Learned:**

**Project 1: Image Captioning** (Easiest - 1h)
- Used BLIP model from Salesforce for image captioning
- Built Streamlit web interface for image upload
- Created image-to-text pipeline

**Project 2: PDF Q&A** (Medium - 2h)
- Processed PDFs and built FAISS index
- Implemented Q&A system with FLAN-T5 model
- Created local RAG system for document querying

**Project 3: Speech-to-Image** (Hardest - 3h)
- Integrated Whisper for speech-to-text conversion
- Combined with Stable Diffusion for text-to-image generation
- Built dual-input system (audio upload or text input)

**Key Concepts:** Multimodal AI, Streamlit, Model integration, Web deployment

---

### **[Week 9: Building Agentic RAG with LlamaIndex](./week9-llamaindex/overview)**
**Files:** `W9_Building_Agentic_RAG_LlamaIndex_3_4.ipynb`

**Prerequisites:** OpenAI API key (paid account)

**What I Have Learned:**

1. **Router Engine** - Routed queries to summary vs vector search engines
2. **Tool Calling** - Implemented LLMs calling functions automatically
3. **Agent Reasoning** - Built multi-step reasoning with FunctionAgent
4. **Multi-Document Agent** - Created agents that query across multiple papers

**Key Concepts:** LlamaIndex framework, Query routing, Tool development, Agent reasoning

---

### **[Week 10: Advanced Agentic RAG](./week10-advanced-rag/overview)**
**Files:** `W10_Building_Agentic_RAG_LlamaIndex_3_4.ipynb`

**What I Have Learned:**
- Built multi-document agent (scaled from 3 papers to 11 papers)
- Implemented tool retrieval system
- Handled complex queries across multiple documents
- Optimized system performance and scaling

**Key Concepts:** Tool retrieval, System scaling, Performance optimization

---

### **[Week 11: Advanced Topics](./week11-advanced-topics/overview)**
**Files:** `W11_L1.ps`, `W11_L2.pdf`

**What I Have Learned:**
- Advanced concepts and topics
- Additional materials and techniques

---

## 🛠️ Technologies I've Used

- **Transformers** (Hugging Face) - Pre-trained models
- **LangChain** - LLM application framework
- **LlamaIndex** - Advanced RAG framework
- **FAISS** - Vector similarity search
- **Streamlit** - Web app framework
- **PyTorch** - Deep learning backend

**Models:** Mistral-7B, BLIP, Whisper, Stable Diffusion, GPT-3.5/GPT-4

---

## 💡 Tips for Success

1. **Start with Week 3** - It's the foundation for everything
2. **Run every cell** - Don't just read, execute and experiment
3. **Understand the why** - Don't copy-paste, learn the concepts
4. **Document your work** - Save prompts, note what works
5. **Build incrementally** - Master basics before advanced topics
6. **Use free resources** - Google Colab GPU, Hugging Face free tier

---

## ⚠️ Common Issues

**"CUDA out of memory"** → Use smaller models or `torch_dtype=torch.float16`

**"Token expired"** → Regenerate token in account settings

**"Module not found"** → Run `!pip install package_name` in Colab

**"API rate limit"** → Add delays: `import time; time.sleep(1)`

---

## 📚 Resources

- [Hugging Face Docs](https://huggingface.co/docs)
- [LangChain Docs](https://python.langchain.com)
- [LlamaIndex Docs](https://docs.llamaindex.ai)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)

---

## ✅ Progress Checklist

- [ ] Week 3: Built RAG system with FAISS
- [ ] Week 6: Completed n8n workflow (optional)
- [ ] Week 7: Completed group project
- [ ] Week 8: Deployed 3 multimodal apps
- [ ] Week 9: Created agentic RAG with LlamaIndex
- [ ] Week 10: Built multi-document agent

---

**Ready to start? Begin with Week 3! 🚀**

> **Source:** This guide is based on the main [README.md](../../README.md) in the project repository.

