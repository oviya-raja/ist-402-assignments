# Week 3 RAG Assignment - System Architecture & Tech Stack

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RAG-BASED QUESTION ANSWERING SYSTEM                    │
│                         Week 3 Assignment Architecture                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           TASK 1: SYSTEM PROMPT                          │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Mistral-7B-Instruct-v0.3                                        │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ System Prompt:                                             │  │   │
│  │  │ "You are a Customer Service Representative for             │  │   │
│  │  │  TechGadgets Online Store..."                              │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TASK 2: GENERATE BUSINESS DATABASE                    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Mistral-7B-Instruct-v0.3                                        │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ Input: System Prompt + Generation Request                 │  │   │
│  │  │ Output: 15 Q&A Pairs                                       │  │   │
│  │  │                                                           │  │   │
│  │  │ Q1: "What is your return policy?"                        │  │   │
│  │  │ A1: "We offer a 30-day return policy..."                 │  │   │
│  │  │ ...                                                       │  │   │
│  │  │ Q15: "Can I leave a product review?"                    │  │   │
│  │  │ A15: "Yes, customers who have purchased..."               │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  TASK 3: FAISS VECTOR DATABASE                           │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: Convert Q&A to Documents                                │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ LangChain Document Objects                                 │  │   │
│  │  │ Document(page_content="Q + A")                             │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: Generate Embeddings                                    │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ sentence-transformers/all-MiniLM-L6-v2                    │  │   │
│  │  │ Input:  Text Document                                     │  │   │
│  │  │ Output: 384-dimensional Vector                             │  │   │
│  │  │                                                           │  │   │
│  │  │ [0.123, -0.456, 0.789, ..., 0.234]  (384 dims)          │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: Store in FAISS Index                                   │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ FAISS Vector Database                                      │  │   │
│  │  │ ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐                  │  │   │
│  │  │ │Vec 1 │ │Vec 2 │ │Vec 3 │ ... │Vec15 │                  │  │   │
│  │  │ └──────┘ └──────┘ └──────┘     └──────┘                  │  │   │
│  │  │ Fast Similarity Search (L2 distance)                      │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TASK 4: CREATE TEST QUESTIONS                         │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Answerable Questions (7)                                       │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ ✓ "What is your return policy?"                           │  │   │
│  │  │ ✓ "How long does shipping take?"                          │  │   │
│  │  │ ✓ "What payment methods do you accept?"                   │  │   │
│  │  │ ...                                                        │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Unanswerable Questions (7)                                     │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ ✗ "What is your company's stock price?"                    │  │   │
│  │  │ ✗ "What is the CEO's email address?"                      │  │   │
│  │  │ ✗ "How many employees work at your company?"              │  │   │
│  │  │ ...                                                        │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TASK 5: RAG PIPELINE TESTING                          │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  RAG Pipeline Function                                           │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ 1. RETRIEVE: FAISS Similarity Search                      │  │   │
│  │  │    Query → Top K Documents                                 │  │   │
│  │  │                                                            │  │   │
│  │  │ 2. AUGMENT: Combine Retrieved Context                     │  │   │
│  │  │    Context = Doc1 + Doc2 + ...                             │  │   │
│  │  │                                                            │  │   │
│  │  │ 3. GENERATE: QA Model Answer                              │  │   │
│  │  │    QA Model(Question, Context) → Answer                   │  │   │
│  │  │                                                            │  │   │
│  │  │ 4. EVALUATE: Confidence Threshold                         │  │   │
│  │  │    if confidence > 0.2: Return Answer                     │  │   │
│  │  │    else: Return "I don't know"                             │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              TASK 6: MODEL EXPERIMENTATION & RANKING                      │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Models Tested (6 Total)                                        │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ 1. consciousAI/question-answering-generative-t5-v1-base    │  │   │
│  │  │ 2. deepset/roberta-base-squad2                            │  │   │
│  │  │ 3. google-bert/bert-large-cased-whole-word-masking        │  │   │
│  │  │ 4. gasolsun/DynamicRAG-8B                                  │  │   │
│  │  │ 5. distilbert-base-uncased-distilled-squad                 │  │   │
│  │  │ 6. mrm8488/bert-base-finetuned-squadv2                     │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                                                                   │   │
│  │  Evaluation Metrics:                                             │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ • Accuracy (Answerable Questions)                          │  │   │
│  │  │ • Confidence Handling (Unanswerable Questions)             │  │   │
│  │  │ • Response Quality                                         │  │   │
│  │  │ • Speed (Latency)                                          │  │   │
│  │  │ • Robustness                                               │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TECHNOLOGY STACK                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  LANGUAGE MODEL LAYER                                                │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ • Mistral-7B-Instruct-v0.3                                     │   │
│  │   └─> System Prompt Design                                    │   │
│  │   └─> Q&A Database Generation                                 │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EMBEDDING LAYER                                                     │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ • sentence-transformers/all-MiniLM-L6-v2                       │   │
│  │   └─> Text → 384-dimensional vectors                          │   │
│  │   └─> Semantic similarity encoding                            │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VECTOR DATABASE LAYER                                               │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ • FAISS (Facebook AI Similarity Search)                       │   │
│  │   └─> Fast similarity search (L2 distance)                     │   │
│  │   └─> Efficient indexing and retrieval                         │   │
│  │   └─> Scalable to millions of vectors                         │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  QA MODEL LAYER                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ • Hugging Face Transformers Pipeline                          │   │
│  │   ├─> distilbert-base-uncased-distilled-squad                 │   │
│  │   ├─> deepset/roberta-base-squad2                             │   │
│  │   ├─> google-bert/bert-large-cased-whole-word-masking         │   │
│  │   ├─> consciousAI/question-answering-generative-t5-v1-base    │   │
│  │   ├─> gasolsun/DynamicRAG-8B                                  │   │
│  │   └─> mrm8488/bert-base-finetuned-squadv2                    │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FRAMEWORK LAYER                                                     │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ • LangChain                                                    │   │
│  │   └─> Document management                                     │   │
│  │   └─> FAISS integration                                       │   │
│  │   └─> Embedding wrappers                                      │   │
│  │                                                                 │   │
│  │ • LangChain Community                                          │   │
│  │   └─> FAISS vector store                                      │   │
│  │   └─> HuggingFace embeddings                                   │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DEEP LEARNING BACKEND                                              │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ • PyTorch                                                      │   │
│  │   └─> Model inference                                          │   │
│  │   └─> Tensor operations                                        │   │
│  │                                                                 │   │
│  │ • Transformers Library                                         │   │
│  │   └─> Pre-trained model access                                │   │
│  │   └─> Pipeline utilities                                       │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
USER QUESTION
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ 1. RETRIEVE (FAISS)                                           │
│    Query Embedding → Similarity Search → Top K Documents      │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ 2. AUGMENT                                                    │
│    Combine Retrieved Documents → Context String              │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ 3. GENERATE (QA Model)                                        │
│    QA Model(Question, Context) → Answer + Confidence Score   │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ 4. EVALUATE                                                   │
│    if confidence > threshold: Return Answer                  │
│    else: Return "I don't know"                               │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
FINAL ANSWER
```

## Package Dependencies

```
┌──────────────────────────────────────────────────────────────┐
│ Core Dependencies                                             │
├──────────────────────────────────────────────────────────────┤
│ • transformers          → Hugging Face models & pipelines    │
│ • torch                 → Deep learning backend             │
│ • sentence-transformers → Text embeddings                    │
│ • faiss-cpu            → Vector similarity search            │
│ • langchain            → LLM application framework           │
│ • langchain-community  → FAISS & embedding integrations     │
│ • pandas               → Data analysis & comparison tables  │
│ • sentencepiece        → Mistral tokenizer                  │
│ • accelerate           → Efficient model loading             │
└──────────────────────────────────────────────────────────────┘
```

## Assignment Requirements Checklist

```
┌──────────────────────────────────────────────────────────────┐
│ ✅ Task 1: System Prompt                                      │
│    └─> Business context defined                              │
│    └─> Role assigned (Customer Service Representative)       │
│    └─> System prompt created                                 │
├──────────────────────────────────────────────────────────────┤
│ ✅ Task 2: Generate Business Database                        │
│    └─> 15 Q&A pairs generated                                │
│    └─> Covers different business aspects                     │
│    └─> Clearly commented in notebook                        │
├──────────────────────────────────────────────────────────────┤
│ ✅ Task 3: FAISS Vector Database                             │
│    └─> Q&A pairs converted to embeddings                      │
│    └─> Stored in FAISS index                                 │
│    └─> Implementation process documented                     │
├──────────────────────────────────────────────────────────────┤
│ ✅ Task 4: Create Test Questions                             │
│    └─> 7 answerable questions                                │
│    └─> 7 unanswerable questions                              │
│    └─> Clearly labeled                                        │
├──────────────────────────────────────────────────────────────┤
│ ✅ Task 5: Test Questions                                    │
│    └─> Answerable questions tested                           │
│    └─> Unanswerable questions tested                         │
│    └─> Results analyzed                                      │
├──────────────────────────────────────────────────────────────┤
│ ✅ Task 6: Model Experimentation                             │
│    └─> 6 models tested (4 required + 2 additional)           │
│    └─> Models ranked with justifications                     │
│    └─> Performance analysis completed                        │
└──────────────────────────────────────────────────────────────┘
```
