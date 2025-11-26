# 1️⃣ Remember (Knowledge)

## Key Definitions Recalled

### Core Concepts

- **RAG (Retrieval-Augmented Generation)**: A technique that combines retrieval of relevant information from a knowledge base with LLM generation to produce accurate, grounded responses
- **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search and clustering of dense vectors, enabling fast retrieval from large vector databases
- **Embeddings**: Vector representations of text that capture semantic meaning, allowing computers to understand relationships between words and concepts
- **Vector Database**: A database that stores data as high-dimensional vectors, enabling semantic similarity search rather than exact keyword matching
- **System Prompt**: Instructions that define the role, behavior, and context for an LLM, shaping how it responds to user queries
- **Similarity Search**: Finding documents or content that are semantically similar to a query, even without exact keyword matches

### Technical Terms

- **Tokenization**: The process of breaking text into smaller units (tokens) that models can process
- **Semantic Search**: Search based on meaning and context rather than exact keyword matching
- **Confidence Score**: A metric indicating how certain a model is about its answer (typically 0-1 scale)
- **Knowledge Base**: A structured collection of information (Q&A pairs, documents) used for retrieval
- **Context Window**: The amount of text a model can process at once
- **Hallucination**: When an LLM generates plausible but incorrect or unsupported information

## Technologies Recalled

### Models and Frameworks

- **Mistral-7B-Instruct-v0.3**: Open-source instruction-tuned language model from Mistral AI, optimized for following instructions and generating structured outputs
- **DistilBERT**: A smaller, faster version of BERT that maintains most of BERT's performance while being more efficient
- **RoBERTa**: An optimized version of BERT with improved training procedures
- **T5 (Text-to-Text Transfer Transformer)**: A model that frames all NLP tasks as text-to-text problems
- **sentence-transformers**: A library for generating sentence embeddings using transformer models
- **all-MiniLM-L6-v2**: A lightweight embedding model that balances speed and quality

### Libraries and Tools

- **LangChain**: Framework for building applications with language models, providing modular components for chains, agents, and memory
- **LangChain Community**: Community-maintained integrations including FAISS vector store support
- **Hugging Face Transformers**: Python library providing access to thousands of pre-trained models
- **PyTorch**: Deep learning framework used as the backend for transformer models
- **FAISS-cpu**: CPU-optimized version of FAISS for vector similarity search

## Key Concepts Recalled

### RAG Pipeline Components

1. **Knowledge Base Creation**: Converting domain-specific information into searchable format
2. **Embedding Generation**: Converting text into numerical vectors that capture semantic meaning
3. **Vector Storage**: Storing embeddings in a database optimized for similarity search
4. **Query Processing**: Converting user questions into embeddings
5. **Retrieval**: Finding most relevant context from the knowledge base
6. **Augmentation**: Combining retrieved context with user query
7. **Generation**: Using LLM to generate final answer based on augmented context

### Prompt Engineering Concepts

- **System Prompts**: Define the AI's role, personality, and constraints
- **Context Injection**: Adding retrieved information to prompts
- **Few-shot Learning**: Providing examples in prompts to guide model behavior
- **Temperature Control**: Adjusting randomness in model outputs

### Evaluation Metrics

- **Accuracy**: How often the model provides correct answers
- **Confidence Score**: Model's certainty about its answer
- **Response Time**: Speed of processing queries
- **Robustness**: Ability to handle edge cases and out-of-scope questions

## Business Context Concepts

- **Domain-Specific Knowledge**: Information specific to a particular business or organization
- **Answerable Questions**: Queries that can be answered from the knowledge base
- **Unanswerable Questions**: Queries requiring information not in the knowledge base
- **Fallback Responses**: Safe responses when confidence is low (e.g., "I don't know")

---

**Reference:** See the [notebook](../../../assignments/W3/W3__QA_Chatbot_Activity_w_Prompt_Engineering%20(1).ipynb) for implementation details.
