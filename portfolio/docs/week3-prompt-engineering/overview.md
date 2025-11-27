# Overview

## Assignment Overview

**Assignment:** RAG-Based Question Answering System with Mistral

**Objective:** Design and implement a Retrieval-Augmented Generation (RAG) system using Mistral-7B-Instruct, FAISS vector database, and custom business data.

## What This Assignment Involved

This assignment required building an intelligent Q&A system that combines:

1. **System Prompt Design** - Creating agentic roles for the LLM (e.g., "You are a marketing expert for a tech startup")
2. **Database Generation** - Using Mistral-7B to generate 10-15 Q&A pairs for a chosen business context
3. **FAISS Implementation** - Converting Q&A pairs into embeddings and storing in a FAISS vector database
4. **Question Testing** - Creating both answerable and unanswerable questions to test system capabilities
5. **Model Comparison** - Testing and ranking multiple Q&A models from Hugging Face

## Objectives Completed

✅ Designed system prompts for LLMs with specific business roles  
✅ Built Q&A databases using Mistral-7B-Instruct  
✅ Implemented FAISS vector database for similarity search  
✅ Created RAG (Retrieval-Augmented Generation) systems  
✅ Compared multiple QA models (DistilBERT, RoBERTa, BERT, T5, DynamicRAG-8B)  
✅ Analyzed model performance on answerable vs. unanswerable questions  
✅ Evaluated confidence scores and output quality  

## Technologies Used

- **Mistral-7B-Instruct-v0.3** - Language model for Q&A generation and system prompts
- **FAISS (faiss-cpu)** - Vector similarity search library for efficient retrieval
- **sentence-transformers** - For creating text embeddings (all-MiniLM-L6-v2 model)
- **Hugging Face Transformers** - Access to pre-trained QA models
- **LangChain** - Framework for building applications with language models
- **LangChain Community** - Extensions for FAISS integration
- **PyTorch** - Deep learning backend

## Models Tested

1. **distilbert-base-uncased-distilled-squad** - Fast, efficient QA model
2. **consciousAI/question-answering-generative-t5-v1-base-s-q-c** - T5-based generative model
3. **deepset/roberta-base-squad2** - RoBERTa-based model
4. **google-bert/bert-large-cased-whole-word-masking-finetuned-squad** - Large BERT model
5. **gasolsun/DynamicRAG-8B** - Advanced RAG model
6. **Additional custom models** - Two additional models of choice

## Assignment Files

- **Main Activity:** `W3__QA_Chatbot_Activity_w_Prompt_Engineering (1).ipynb`
- **Location:** [`../../assignments/W3/`](../../../assignments/W3/) directory

## Key Learning Outcomes

This assignment demonstrated:
- How to design effective system prompts for LLMs
- How to generate domain-specific knowledge bases using LLMs
- How to implement vector databases for semantic search
- How to build end-to-end RAG pipelines
- How to evaluate and compare multiple AI models
- How to handle both answerable and unanswerable questions

---

## Next Steps

Document your learnings using Bloom's Taxonomy:

- [1️⃣ Remember](./remember) - Key concepts and definitions
- [2️⃣ Understand](./understand) - How RAG systems work
- [3️⃣ Apply](./apply) - Implementation steps and code
- [4️⃣ Analyze](./analyze) - Model comparisons and breakdowns
- [5️⃣ Evaluate](./evaluate) - Performance assessment and judgments
- [6️⃣ Create](./create) - Final deliverables and original work

**📖 Need help documenting?** See the [Documentation Guide](../documentation-guide) for detailed instructions.
