#!/usr/bin/env python3
"""
QA Chatbot Activity - RAG-Based Question Answering System

This script implements a simple RAG (Retrieval-Augmented Generation) chatbot using:
- LangChain for document management and FAISS integration
- HuggingFace embeddings for text vectorization
- Transformers pipeline for question-answering
- FAISS for efficient similarity search

The system:
1. Creates a knowledge base from FAQ data
2. Converts FAQ data into LangChain Document objects
3. Creates embeddings using HuggingFace models
4. Builds a FAISS vector database for similarity search
5. Loads a QA model for answer generation
6. Tests the RAG system with sample questions
"""

import sys
import os
import warnings
from typing import List, Tuple, Any, Dict

# Suppress verbose warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights.*not initialized.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN.*")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import required libraries
try:
    from transformers import pipeline
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    print("✅ All required libraries imported successfully!")
except ImportError as e:
    print("❌ Required packages not installed!")
    print("   Install with: pip install transformers langchain langchain-community sentence-transformers faiss-cpu torch")
    print(f"   Error: {e}")
    sys.exit(1)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_ID = "distilbert-base-uncased-distilled-squad"
CONFIDENCE_THRESHOLD = 0.2  # Minimum confidence score for accepting an answer
SIMILARITY_SEARCH_K = 2  # Number of top documents to retrieve


# ============================================================================
# STEP 1: SETUP KNOWLEDGE BASE
# ============================================================================

def create_faq_data() -> List[Tuple[str, str]]:
    """
    Create a simple knowledge base with question-answer pairs.
    This is like creating a mini-encyclopedia for our chatbot.
    
    Returns:
        List of (question, answer) tuples
    """
    print("=" * 60)
    print("STEP 1: Setting Up Knowledge Base")
    print("=" * 60)
    print("\n📚 Creating FAQ knowledge base...")
    
    faq_data = [
        ("What is Python?", "Python is a high-level programming language known for its simplicity and readability."),
        ("What is machine learning?", "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed."),
        ("What is a chatbot?", "A chatbot is a computer program designed to simulate conversation with human users."),
        ("What is the return policy?", "30 days return with full refund."),
        ("What are your store hours?", "We are open 9am–9pm, Mon–Sat."),
        ("Do you ship internationally?", "Yes, we ship worldwide, including Australia.")
    ]
    
    print(f"✅ Created {len(faq_data)} FAQ pairs\n")
    return faq_data


# ============================================================================
# STEP 2: CONVERT TO LANGCHAIN DOCUMENTS
# ============================================================================

def create_documents(faq_data: List[Tuple[str, str]]) -> List[Document]:
    """
    Convert FAQ data into LangChain Document objects.
    Each document contains both the question and answer as searchable content.
    
    Args:
        faq_data: List of (question, answer) tuples
    
    Returns:
        List of LangChain Document objects
    """
    print("=" * 60)
    print("STEP 2: Converting to LangChain Documents")
    print("=" * 60)
    print("\n📚 Converting FAQ data to Document objects...")
    print("   Each document contains question + answer as searchable content")
    
    # List comprehension: [expression for item in list] creates a new list
    documents = [Document(page_content=qa[0] + " " + qa[1]) for qa in faq_data]
    
    print(f"✅ Created {len(documents)} Document objects\n")
    return documents


# ============================================================================
# STEP 3: CREATE EMBEDDINGS MODEL
# ============================================================================

def create_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Create embeddings model - this converts text into numerical vectors.
    We use a pre-trained model that's good at understanding sentence meanings.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    print("=" * 60)
    print("STEP 3: Creating Embeddings Model")
    print("=" * 60)
    print(f"\n📚 Loading embedding model: {EMBEDDING_MODEL_ID}")
    print("   This converts text to numerical vectors for similarity search...")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    
    print("✅ Embedding model loaded!\n")
    return embeddings


# ============================================================================
# STEP 4: CREATE FAISS DATABASE
# ============================================================================

def create_faiss_database(documents: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Create a FAISS database from documents.
    FAISS allows us to quickly find the most relevant documents for any question.
    
    Args:
        documents: List of LangChain Document objects
        embeddings: HuggingFaceEmbeddings instance
    
    Returns:
        FAISS vector store
    """
    print("=" * 60)
    print("STEP 4: Creating FAISS Vector Database")
    print("=" * 60)
    print("\n📚 Building FAISS index from documents...")
    print("   This enables fast similarity search (milliseconds)")
    
    # Create FAISS database from documents
    db = FAISS.from_documents(documents, embeddings)
    
    print(f"✅ FAISS database created with {len(documents)} documents\n")
    return db


# ============================================================================
# STEP 5: LOAD QA MODEL
# ============================================================================

def load_qa_pipeline() -> Any:
    """
    Load a pre-trained question-answering model.
    DistilBERT is a smaller, faster version of BERT that's good for Q&A tasks.
    
    Returns:
        QA pipeline instance
    
    Note:
        Additional models you can try:
        - consciousAI/question-answering-generative-t5-v1-base-s-q-c
        - deepset/roberta-base-squad2
        - google-bert/bert-large-cased-whole-word-masking-finetuned-squad
        - gasolsun/DynamicRAG-8B
    """
    print("=" * 60)
    print("STEP 5: Loading Question-Answering Model")
    print("=" * 60)
    print(f"\n📚 Loading QA model: {QA_MODEL_ID}")
    print("   DistilBERT is a smaller, faster version of BERT")
    print("   Good for Q&A tasks with fast inference...")
    
    qa_pipeline = pipeline("question-answering", model=QA_MODEL_ID)
    
    print("✅ QA pipeline loaded successfully!\n")
    return qa_pipeline


# ============================================================================
# STEP 6: CREATE TEST QUESTIONS
# ============================================================================

def create_test_questions() -> List[str]:
    """
    Create test questions to evaluate the RAG system.
    
    Returns:
        List of question strings
    """
    print("=" * 60)
    print("STEP 6: Creating Test Questions")
    print("=" * 60)
    print("\n📚 Creating test questions...")
    
    questions = [
        "Can I return a product after 2 weeks?",
        "Do you ship to Australia?",
        "What time do you open on Monday?",
        "Do you sell electronics?",
        "what is python"
    ]
    
    print(f"✅ Created {len(questions)} test questions\n")
    return questions


# ============================================================================
# STEP 7: TEST RAG SYSTEM
# ============================================================================

def test_rag_system(
    questions: List[str],
    db: FAISS,
    qa_pipeline: Any,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    k: int = SIMILARITY_SEARCH_K
) -> None:
    """
    Test the RAG system with questions.
    
    Process:
    1. Find relevant information using FAISS similarity search
    2. Prepare context for the AI model
    3. Generate answer using QA model
    4. Confidence-based answer selection
    5. Display results
    
    Args:
        questions: List of questions to test
        db: FAISS vector database
        qa_pipeline: QA pipeline instance
        confidence_threshold: Minimum confidence score to accept answer
        k: Number of top documents to retrieve
    """
    print("=" * 60)
    print("STEP 7: Testing RAG System")
    print("=" * 60)
    print(f"\n📊 Confidence threshold: {confidence_threshold}")
    print(f"📊 Retrieving top {k} documents per question")
    print("\n--- Week 2 Chatbot Response ---\n")
    
    for q in questions:
        # STEP 1: FIND RELEVANT INFORMATION
        # Use FAISS similarity search to find the k most relevant FAQ documents
        # 'db' is our FAISS vector database containing embedded FAQ data
        # 'k' means "return the top k most similar documents"
        docs = db.similarity_search(q, k=k)
        
        # STEP 2: PREPARE CONTEXT FOR THE AI MODEL
        # Combine the content from retrieved documents into one text string
        # This creates the "context" that will help the QA model answer accurately
        # Each 'd.page_content' contains the text from a retrieved FAQ document
        context = " ".join([d.page_content for d in docs])
        
        # STEP 3: GENERATE ANSWER USING QA MODEL
        # Pass both the user's question AND the retrieved context to the QA pipeline
        # The model will use the context to generate a more accurate, grounded answer
        # This is the "Augmented" part of Retrieval-Augmented Generation (RAG)
        result = qa_pipeline({"question": q, "context": context})
        
        # STEP 4: CONFIDENCE-BASED ANSWER SELECTION
        # Check if the model is confident enough in its answer (score > threshold)
        # If confidence is too low, use a safe fallback response
        # This prevents the chatbot from giving unreliable or hallucinated answers
        answer = result["answer"] if result["score"] > confidence_threshold else "I don't know."
        
        # STEP 5: DISPLAY RESULTS
        # Format and print the question-answer pair for easy reading
        # Also show confidence score for educational purposes
        print(f"Q: {q}")
        print(f"A: {answer}")
        if result["score"] > confidence_threshold:
            print(f"   (Confidence: {result['score']:.3f})")
        else:
            print(f"   (Confidence too low: {result['score']:.3f} < {confidence_threshold})")
        print()


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main() -> None:
    """Main orchestration function."""
    print("=" * 60)
    print("QA Chatbot Activity - RAG-Based Question Answering System")
    print("=" * 60)
    print("\nThis script demonstrates a simple RAG chatbot using:")
    print("  - LangChain for document management")
    print("  - FAISS for vector similarity search")
    print("  - HuggingFace embeddings for text vectorization")
    print("  - Transformers pipeline for question-answering")
    print()
    
    try:
        # Step 1: Setup knowledge base
        faq_data = create_faq_data()
        
        # Step 2: Convert to LangChain Documents
        documents = create_documents(faq_data)
        
        # Step 3: Create embeddings model
        embeddings = create_embeddings_model()
        
        # Step 4: Create FAISS database
        db = create_faiss_database(documents, embeddings)
        
        # Step 5: Load QA model
        qa_pipeline = load_qa_pipeline()
        
        print("=" * 60)
        print("✅ All Components Loaded Successfully!")
        print("=" * 60)
        print()
        
        # Step 6: Create test questions
        questions = create_test_questions()
        
        # Step 7: Test RAG system
        test_rag_system(questions, db, qa_pipeline)
        
        print("=" * 60)
        print("✅ QA Chatbot Activity Completed!")
        print("=" * 60)
        print()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

