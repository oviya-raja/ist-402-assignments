# Week 3 Assignment: RAG-Based Question Answering System with Mistral

## Assignment Overview

**Course:** IST402 - AI Agents & RAG Systems  
**Week:** 3  
**Topic:** Prompt Engineering & QA Applications  
**Assignment Type:** Class Activity

---

## Objective

Students will design and implement a **Retrieval-Augmented Generation (RAG)** system using Mistral-7B-Instruct, FAISS vector database, and custom business data.

**Submission:** Submit the link to your completed notebook.

---

## Instructions

### **1. Create an Assistant System Prompt**

- Using `mistralai/Mistral-7B-Instruct-v0.3`, design a system prompt that gives the model a specific role
- Examples:
  - "You are a marketing expert for a tech startup"
  - "You are a database creator for a healthcare organization"
- Choose a specific organization/business context you'll work with throughout this activity

**Deliverable:** A well-designed system prompt that defines the AI's role and business context.

---

### **2. Generate Business Database Content**

- Use `mistralai/Mistral-7B-Instruct-v0.3` from Hugging Face to create prompts that generate:
  - A comprehensive Q&A database for your chosen business/organization
  - **Minimum 10-15 question-answer pairs** covering different aspects of the business
- **Add comments in your notebook clearly showing the new database Q&A pairs**

**Deliverable:** A complete Q&A database with 10-15 pairs, clearly commented in the notebook.

---

### **3. Implement FAISS Vector Database**

- Convert your generated Q&A database into embeddings
- Store the embeddings in a FAISS index for efficient similarity search
- **Use comments to demonstrate the database implementation process**

**Deliverable:** Working FAISS vector database with embedded Q&A pairs.

---

### **4. Create Test Questions**

- Using `mistralai/Mistral-7B-Instruct-v0.3`, generate two types of questions:
  - **Answerable questions**: Can be directly answered from your database
    - Create at least **5 answerable questions**
  - **Unanswerable questions**: Require information not present in your database
    - Create at least **5 unanswerable questions**

**Deliverable:** Two clearly labeled sets of test questions (answerable and unanswerable).

---

### **5. Implement and Test Questions**

- Run both types of questions through your RAG system
- **Use clear comments to differentiate between:**
  - Questions that can be answered (with expected good retrieval)
  - Questions that cannot be answered (testing system limitations)

**Deliverable:** Test results showing how the system handles both question types.

---

### **6. Model Experimentation and Ranking**

- Test your RAG system with multiple Q&A models from Hugging Face
- **Experiment with these specific QA models:**

```python
"""Required models to test"""
qa_pipeline = pipeline("question-answering", model="consciousAI/question-answering-generative-t5-v1-base-s-q-c")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
qa_pipeline = pipeline("question-answering", model="google-bert/bert-large-cased-whole-word-masking-finetuned-squad")
qa_pipeline = pipeline("question-answering", model="gasolsun/DynamicRAG-8B")
```

- **Add two additional QA models of your choice** from Hugging Face
- **Test all models with both answerable and unanswerable questions**
- **Rank the models from best to worst performance and explain why**
- **Identify which model(s) provide confidence scores but still show reasonable output**
- **Compare performance across different question types (factual, reasoning, out-of-scope)**

#### **Model Evaluation Criteria:**

- **Accuracy**: How well does it answer questions from your database?
- **Confidence Handling**: Does it appropriately indicate uncertainty for unanswerable questions?
- **Response Quality**: Are the answers coherent and relevant?
- **Speed**: How fast does each model process queries?
- **Robustness**: How does it handle edge cases and out-of-scope questions?

**Deliverable:**

- Comparison table/matrix of all 6 models
- Ranking with justifications
- Performance analysis across different question types

---

## Technical Requirements

### Installation

```python
# Required packages
%pip install transformers torch sentence-transformers faiss-cpu langchain langchain-community
```

### Key Components to Include

1. **System Prompt Design** - Clear agentic role definition
2. **Database Generation** - Mistral-generated business Q&A pairs
3. **FAISS Implementation** - Vector storage and retrieval
4. **Question Testing** - Both answerable and unanswerable queries
5. **Model Comparison** - Performance analysis and ranking
6. **Confidence Analysis** - Model uncertainty vs. output quality

---

## Deliverables Checklist

Your notebook should demonstrate:

- [ ] **Business Context**: Clear organization/role you've chosen
- [ ] **Generated Database**: Commented Q&A pairs created by Mistral (10-15 pairs)
- [ ] **FAISS Integration**: Working vector database implementation
- [ ] **Question Analysis**: Clear separation of answerable vs. unanswerable questions (5+ each)
- [ ] **Model Evaluation**: Ranked comparison of 6 different Q&A models
- [ ] **Performance Insights**: Analysis of confidence scores and output quality
- [ ] **Reflection**: Strengths, weaknesses, and real-world applications
- [ ] **Clear Documentation**: Meaningful comments throughout the notebook

---

## Evaluation Criteria

Your assignment will be evaluated on:

- **Creativity** in business context and agentic role design
- **Technical Implementation** of RAG pipeline with FAISS
- **Quality Analysis** of different Q&A models
- **Clear Documentation** with meaningful comments
- **Critical Thinking** in model comparison and limitations analysis

---

## Important Notes

- Focus on building a system that could realistically be deployed for a real business use case
- Consider scalability, accuracy, and user experience in your implementation choices
- Document your thought process and decision-making throughout the notebook
- Test thoroughly with both answerable and unanswerable questions
- Provide clear justifications for your model rankings

---

## Resources

- **Mistral-7B-Instruct**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Hugging Face Models**: https://huggingface.co/models?pipeline_tag=question-answering
- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **LangChain Documentation**: https://python.langchain.com

---

**Good luck with your assignment!** 🚀
