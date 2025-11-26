# 3️⃣ Apply (Application)

## Implementation Steps

### Step 1: Install Required Libraries

I applied the following installation commands:

```python
%pip install transformers          # For pre-trained AI models (BERT, DistilBERT, etc.)
%pip install langchain             # Framework for building applications with language models
%pip install langchain-community   # Community extensions for LangChain
%pip install sentence-transformers # For creating text embeddings (converting text to numbers)
%pip install torch                  # PyTorch - deep learning framework (backend for transformers)
%pip install faiss-cpu             # Facebook AI Similarity Search - for fast similarity searches
```

### Step 2: Import Libraries and Setup

I applied these imports to access the necessary functionality:

```python
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
```

### Step 3: Create Knowledge Base

I applied list comprehension to structure FAQ data:

```python
faq_data = [
    ("What is Python?", "Python is a high-level programming language known for its simplicity and readability."),
    ("What is machine learning?", "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed."),
    ("What is a chatbot?", "A chatbot is a computer program designed to simulate conversation with human users."),
    ("What is the return policy?", "30 days return with full refund."),
    ("What are your store hours?", "We are open 9am–9pm, Mon–Sat."),
    ("Do you ship internationally?", "Yes, we ship worldwide, including Australia.")
]
```

### Step 4: Convert to LangChain Documents

I applied list comprehension to convert FAQ pairs into Document objects:

```python
documents = [Document(page_content=qa[0] + " " + qa[1]) for qa in faq_data]
```

This combines question and answer into searchable content for each document.

### Step 5: Create Embeddings Model

I applied HuggingFaceEmbeddings to create the embedding model:

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

This model converts text into 384-dimensional vectors that capture semantic meaning.

### Step 6: Build FAISS Vector Database

I applied FAISS.from_documents() to create the vector index:

```python
db = FAISS.from_documents(documents, embeddings)
```

This automatically:
- Converts all documents to embeddings
- Creates an optimized index structure
- Enables fast similarity search

### Step 7: Load QA Model

I applied the pipeline function to load a pre-trained QA model:

```python
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
```

Alternative models tested:
```python
# qa_pipeline = pipeline("question-answering", model="consciousAI/question-answering-generative-t5-v1-base-s-q-c")
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
# qa_pipeline = pipeline("question-answering", model="google-bert/bert-large-cased-whole-word-masking-finetuned-squad")
# qa_pipeline = pipeline("question-answering", model="gasolsun/DynamicRAG-8B")
```

### Step 8: Implement RAG Pipeline

I applied the complete RAG workflow:

```python
# Define test questions
questions = [
    "Can I return a product after 2 weeks?",
    "Do you ship to Australia?",
    "What time do you open on Monday?",
    "Do you sell electronics?",
    "what is python"
]

# Process each question
for q in questions:
    # STEP 1: FIND RELEVANT INFORMATION
    docs = db.similarity_search(q, k=2)  # Retrieve top 2 most similar documents
    
    # STEP 2: PREPARE CONTEXT FOR THE AI MODEL
    context = " ".join([d.page_content for d in docs])  # Combine retrieved documents
    
    # STEP 3: GENERATE ANSWER USING QA MODEL
    result = qa_pipeline({"question": q, "context": context})
    
    # STEP 4: CONFIDENCE-BASED ANSWER SELECTION
    answer = result["answer"] if result["score"] > 0.2 else "I don't know."
    
    # STEP 5: DISPLAY RESULTS
    print(f"\nQ: {q}\nA: {answer}")
```

## Key Code Patterns Applied

### Pattern 1: Similarity Search
```python
docs = db.similarity_search(query, k=2)
```
- `k=2` retrieves the top 2 most relevant documents
- Returns Document objects with page_content and metadata

### Pattern 2: Context Augmentation
```python
context = " ".join([d.page_content for d in docs])
```
- Combines multiple retrieved documents into single context string
- Provides comprehensive background for the QA model

### Pattern 3: Confidence Threshold
```python
answer = result["answer"] if result["score"] > 0.2 else "I don't know."
```
- Uses conditional logic to handle low-confidence answers
- Prevents hallucination by falling back to safe response

### Pattern 4: Pipeline Pattern
```python
result = qa_pipeline({"question": q, "context": context})
```
- Uses Hugging Face pipeline abstraction
- Automatically handles tokenization, model inference, and post-processing

## Applied Techniques

### 1. System Prompt Design
- Created prompts defining specific business roles
- Used Mistral-7B to generate domain-specific Q&A pairs
- Structured prompts to guide model behavior

### 2. Vector Database Management
- Converted text to embeddings automatically
- Stored embeddings in optimized FAISS index
- Performed similarity search efficiently

### 3. Model Comparison Framework
- Tested multiple QA models systematically
- Compared performance on same question set
- Evaluated confidence scores and response quality

### 4. Error Handling
- Implemented confidence-based fallback
- Handled unanswerable questions gracefully
- Prevented hallucination with threshold logic

## Implementation Challenges Solved

1. **Embedding Model Selection**: Chose all-MiniLM-L6-v2 for balance of speed and quality
2. **Retrieval Count**: Set k=2 to get enough context without overwhelming the model
3. **Confidence Threshold**: Used 0.2 as threshold based on testing different values
4. **Context Formatting**: Combined documents with space separator for clean context

---

**Reference:** See the [notebook](../../../assignments/W3/W3__QA_Chatbot_Activity_w_Prompt_Engineering%20(1).ipynb) for complete implementation.
