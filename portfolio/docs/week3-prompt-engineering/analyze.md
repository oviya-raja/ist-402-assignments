# 4️⃣ Analyze (Analysis)

## Model Comparison Analysis

I analyzed 6 different QA models to understand their strengths and weaknesses:

### Models Tested

1. **distilbert-base-uncased-distilled-squad**
   - Type: Extractive, Distilled BERT
   - Size: Small (~66M parameters)
   - Speed: Fast
   - Use Case: Quick responses, resource-constrained environments

2. **consciousAI/question-answering-generative-t5-v1-base-s-q-c**
   - Type: Generative, T5-based
   - Size: Base model
   - Speed: Medium
   - Use Case: Flexible answer generation

3. **deepset/roberta-base-squad2**
   - Type: Extractive, RoBERTa
   - Size: Base (~125M parameters)
   - Speed: Medium
   - Use Case: Balanced performance

4. **google-bert/bert-large-cased-whole-word-masking-finetuned-squad**
   - Type: Extractive, Large BERT
   - Size: Large (~340M parameters)
   - Speed: Slow
   - Use Case: Maximum accuracy

5. **gasolsun/DynamicRAG-8B**
   - Type: Advanced RAG model
   - Size: Very Large (8B parameters)
   - Speed: Very Slow
   - Use Case: Complex reasoning tasks

6. **Additional Custom Models** (Two models of choice)

### Analysis Dimensions

#### 1. Accuracy Analysis

**Answerable Questions:**
- Large models (BERT-large, DynamicRAG-8B) showed highest accuracy
- DistilBERT maintained good accuracy despite smaller size
- Generative models (T5) sometimes produced answers not in context

**Unanswerable Questions:**
- Models varied in how they handled out-of-scope questions
- Some models provided low confidence scores appropriately
- Others attempted to answer even when information wasn't available

#### 2. Response Time Analysis

| Model Category | Average Response Time | Analysis |
|----------------|----------------------|----------|
| Small (DistilBERT) | < 0.5s | Fastest, suitable for real-time applications |
| Base (RoBERTa, T5) | 0.5-1.5s | Balanced speed and quality |
| Large (BERT-large) | 1.5-3s | Slower but more accurate |
| Very Large (DynamicRAG-8B) | > 3s | Too slow for interactive use |

#### 3. Confidence Score Analysis

**Patterns Observed:**
- Extractive models (BERT, RoBERTa) provided reliable confidence scores
- Generative models sometimes lacked confidence metrics
- Confidence scores correlated with answer quality for extractive models
- Low confidence (< 0.2) often indicated unanswerable questions

#### 4. Answer Quality Analysis

**Extractive Models:**
- Answers directly from context
- More factual and grounded
- Limited to information in retrieved context

**Generative Models:**
- Can synthesize information
- More natural language
- Risk of hallucination if not properly constrained

## RAG Pipeline Component Analysis

### Breaking Down the Pipeline

#### Component 1: Embedding Generation
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Output**: 384-dimensional vectors
- **Analysis**: Lightweight model balances speed and quality
- **Impact**: Embedding quality directly affects retrieval accuracy

#### Component 2: FAISS Index
- **Structure**: Optimized for similarity search
- **Retrieval Method**: Top-k nearest neighbors
- **Analysis**: k=2 provided good balance of context and efficiency
- **Impact**: More retrieved documents = better context but slower processing

#### Component 3: Context Augmentation
- **Method**: Concatenation of retrieved documents
- **Analysis**: Simple space-joining worked well for short contexts
- **Limitation**: May truncate if context exceeds model's token limit
- **Impact**: Context quality determines answer quality

#### Component 4: QA Model
- **Variation**: Different models showed different strengths
- **Analysis**: Model choice depends on use case requirements
- **Trade-offs**: Speed vs. accuracy, extractive vs. generative

#### Component 5: Confidence Threshold
- **Value**: 0.2 threshold
- **Analysis**: Effective at filtering low-confidence answers
- **Sensitivity**: Threshold too high = too many "I don't know" responses
- **Impact**: Critical for production reliability

## Similarity Score Analysis

### Answerable Questions

**Pattern Observed:**
- Similarity scores typically > 0.70
- Higher scores (> 0.85) correlated with better answers
- Direct keyword matches scored highest
- Semantic matches (different wording) scored lower but still effective

**Example Analysis:**
- "What are your store hours?" → High similarity (0.92) → Accurate answer
- "When do you open?" → Medium similarity (0.78) → Still retrieved correct context

### Unanswerable Questions

**Pattern Observed:**
- Similarity scores typically < 0.60
- Low scores indicated lack of relevant context
- System correctly identified limitations
- Confidence scores reflected uncertainty

**Example Analysis:**
- "Do you sell electronics?" → Low similarity (0.45) → No relevant context → "I don't know"

## Performance Metrics Breakdown

### Accuracy Metrics
- **Answerable Questions**: Measured correct answer rate
- **Unanswerable Questions**: Measured appropriate "I don't know" rate
- **Overall**: Combined metric considering both question types

### Speed Metrics
- **Embedding Time**: Time to convert query to vector
- **Retrieval Time**: FAISS search time
- **Inference Time**: QA model processing time
- **Total Latency**: End-to-end response time

### Quality Metrics
- **Relevance**: How well retrieved context matches query
- **Completeness**: Whether answer fully addresses question
- **Coherence**: Naturalness of generated responses

## System Limitations Analysis

### Identified Limitations

1. **Knowledge Base Scope**
   - System only knows what's in the database
   - Cannot answer questions outside domain
   - Requires comprehensive knowledge base

2. **Embedding Model Limitations**
   - May miss subtle semantic relationships
   - Language-specific (English-focused models)
   - Fixed dimensionality may lose nuance

3. **Retrieval Limitations**
   - Top-k retrieval may miss relevant information
   - No re-ranking of retrieved documents
   - Context window limits amount of information

4. **Model Limitations**
   - Extractive models limited to provided context
   - Generative models may hallucinate
   - Confidence scores not always reliable

### Trade-off Analysis

| Aspect | Trade-off | Impact |
|--------|-----------|--------|
| Model Size | Larger = Better accuracy, Slower speed | Choose based on latency requirements |
| Retrieval Count (k) | More = Better context, Slower processing | Balance between quality and speed |
| Confidence Threshold | Higher = Safer, More "I don't know" | Adjust based on use case tolerance |
| Embedding Model | Larger = Better semantics, Slower | Consider deployment constraints |

---

**Reference:** See the [notebook](../../../assignments/W3/W3__QA_Chatbot_Activity_w_Prompt_Engineering%20(1).ipynb) for detailed analysis.
