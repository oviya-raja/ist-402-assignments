# 2️⃣ Understand (Comprehension)

## How RAG Systems Work

I understand that RAG (Retrieval-Augmented Generation) works in a three-stage pipeline:

### 1. Retrieve Stage
- User query is converted into an embedding (numerical vector) using a sentence transformer model
- The embedding is compared against all stored embeddings in the FAISS vector database
- FAISS uses efficient algorithms (like L2 distance or cosine similarity) to find the most similar documents
- Top-k most relevant documents are retrieved (typically k=2-5 for Q&A systems)

### 2. Augment Stage
- Retrieved documents are combined into a single context string
- This context is prepended or appended to the user's original question
- The augmented prompt now contains both the question and relevant background information
- This gives the LLM the necessary context to answer accurately

### 3. Generate Stage
- The augmented prompt is passed to a question-answering model (like DistilBERT)
- The model processes both the question and context together
- It extracts or generates an answer based on the provided context
- A confidence score indicates how certain the model is about its answer

## Why Vector Embeddings Enable Semantic Search

I understand that embeddings work by:

- **Capturing Semantic Meaning**: Words with similar meanings are positioned close together in vector space
- **Dimensionality**: Each word/sentence becomes a high-dimensional vector (e.g., 384 dimensions for all-MiniLM-L6-v2)
- **Mathematical Relationships**: Similarity is measured using distance metrics (cosine similarity, Euclidean distance)
- **Context Awareness**: Embeddings consider surrounding words, not just individual terms
- **Cross-Language Capabilities**: Similar concepts in different languages can be close in vector space

**Example**: "What are your store hours?" and "When do you open?" would have similar embeddings even though they use different words.

## How FAISS Enables Fast Similarity Search

I understand that FAISS is effective because:

- **Index Structures**: Uses specialized data structures (IVF, HNSW) optimized for similarity search
- **Approximate Search**: Can trade some accuracy for significant speed improvements
- **Scalability**: Handles millions to billions of vectors efficiently
- **Memory Efficiency**: Compresses vectors while maintaining search quality
- **GPU Support**: Can leverage GPU acceleration for even faster searches

The key insight: Instead of comparing a query against every document (O(n) complexity), FAISS uses indexing to reduce this to O(log n) or better.

## Why System Prompts Shape LLM Behavior

I understand that system prompts work by:

- **Role Definition**: Telling the model "You are a marketing expert" sets expectations for expertise level
- **Context Setting**: Providing business context helps the model generate domain-appropriate responses
- **Constraint Setting**: Defining boundaries (e.g., "only answer from provided context") prevents hallucination
- **Tone Control**: Instructions like "be professional" or "be friendly" influence response style
- **Output Format**: Specifying structure (e.g., "format as Q&A pairs") guides generation

**Key Insight**: System prompts act as instructions that persist throughout the conversation, unlike user messages which are transient.

## How Confidence Scores Work

I understand confidence scores by:

- **Probability Distribution**: Models output probability distributions over possible answers
- **Score Calculation**: Confidence = probability of the selected answer relative to alternatives
- **Threshold Setting**: Low scores (<0.2) indicate uncertainty, triggering fallback responses
- **Model-Specific**: Different models use different scoring methods (some provide scores, others don't)
- **Quality Indicator**: Higher scores generally (but not always) correlate with better answers

## Why Testing Both Answerable and Unanswerable Questions Matters

I understand this is important because:

- **System Boundaries**: Unanswerable questions test whether the system correctly identifies its limitations
- **Hallucination Detection**: Good systems should say "I don't know" rather than make up answers
- **Confidence Calibration**: Models should show low confidence for unanswerable questions
- **Real-World Readiness**: Production systems must handle questions outside their knowledge base
- **Evaluation Completeness**: Only testing answerable questions gives an incomplete picture

## How Different QA Models Work

I understand the differences:

- **Extractive Models** (DistilBERT, RoBERTa, BERT): Extract answers directly from provided context
- **Generative Models** (T5, DynamicRAG-8B): Generate answers that may not appear verbatim in context
- **Model Size Trade-offs**: Larger models (BERT-large) are more accurate but slower
- **Training Differences**: Models trained on SQuAD dataset are optimized for Q&A tasks
- **Architecture Impact**: Transformer architecture variations affect performance characteristics

## The Relationship Between Components

I understand how everything connects:

```
User Question
    ↓
Embedding Model (sentence-transformers)
    ↓
FAISS Vector Search
    ↓
Retrieved Context
    ↓
Augmented Prompt (Question + Context)
    ↓
QA Model (DistilBERT, etc.)
    ↓
Answer + Confidence Score
    ↓
Fallback Logic (if confidence < threshold)
    ↓
Final Response
```

Each component depends on the previous one, and the quality of each stage affects the final output.

---

**Reference:** See the [notebook](../../../assignments/W3/W3__QA_Chatbot_Activity_w_Prompt_Engineering%20(1).ipynb) for implementation details.
