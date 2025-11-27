#!/usr/bin/env python3
"""
RAG System Exercise - Building a Complete RAG System with Mistral

This script implements a complete RAG (Retrieval-Augmented Generation) system:
- Step 1: Create business-specific system prompts
- Step 2: Generate Q&A database using Mistral
- Step 3: Implement FAISS vector database for similarity search
- Step 4: Create test questions (answerable and unanswerable)
- Step 5: Test RAG system performance
- Step 6: Evaluate and rank multiple QA models

Performance Notes:
- Uses same device optimization as prompt_engineering_basics.py
- FAISS provides fast similarity search (milliseconds)
- Model evaluation includes speed, confidence, and quality metrics
"""

import sys
import os
import time
import json
import re
import warnings
from typing import Dict, Tuple, List, Any, Optional

# Suppress verbose warnings for cleaner student output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights.*not initialized.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN.*")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Optional dependency

# Import required libraries
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging
    from sentence_transformers import SentenceTransformer
    import torch
    import numpy as np
    import faiss
    
    # Suppress transformers library warnings
    transformers_logging.set_verbosity_error()
except (ImportError, RuntimeError) as e:
    error_msg = str(e)
    if "register_fake" in error_msg or "torch.library" in error_msg:
        print("❌ Dependency version mismatch detected!")
        print("\n🔧 Fix: Update PyTorch and torchvision to compatible versions:")
        print("   pip install --upgrade torch torchvision")
        print("\n   Or install specific compatible versions:")
        print("   pip install torch>=2.0.0 torchvision>=0.15.0")
        print("\n💡 This error occurs when torchvision requires features not in your current torch version.")
    elif "transformers" in error_msg.lower() or "ImportError" in str(type(e)):
        print("❌ Required packages not installed!")
        print("   Install with: pip install transformers torch sentence-transformers faiss-cpu")
    else:
        print(f"❌ Error importing required libraries: {error_msg}")
        print("\n🔧 Try updating packages:")
        print("   pip install --upgrade transformers torch torchvision sentence-transformers faiss-cpu")
    sys.exit(1)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight and fast

# ============================================================================
# QA MODEL CONFIGURATION
# ============================================================================
# 
# Extractive vs Generative Models:
# - Extractive: Finds answer directly in context (fast, reliable, has confidence scores) - Best for RAG systems
# - Generative: Creates new text to answer (slower, more flexible, no confidence scores) - Better for creative tasks
# 
# Why Explicit QA Models?
# -----------------------
# Explicit QA models are specifically trained for question-answering tasks.
# They provide:
# - Confidence scores: Tell you how sure the model is about its answer
# - Better accuracy: Trained specifically on QA datasets (SQuAD, etc.)
# - Faster inference: Optimized for QA, not general text generation
# - Structured output: Returns answer + confidence + start/end positions
#
# General models (like Qwen) can do QA but:
# - No confidence scores (harder to evaluate)
# - Slower (not optimized for QA)
# - Less accurate (not specifically trained for QA)
#
# Performance Metrics We'll Compare:
# - Accuracy: How correct are the answers?
# - Speed: How fast is inference? (milliseconds)
# - Confidence Scores: How sure is the model? (0.0 to 1.0)
# - Composite Score: Balanced metric combining all factors
#
# Model Types:
# - Extractive: Finds answer span in context (faster, more reliable)
# - Generative: Generates new text (slower, more creative but less reliable)
# ============================================================================

QA_MODELS = [
    # ========================================================================
    # REQUIRED MODELS (4) - From class exercise
    # ========================================================================
    
    # 1. T5-based Generative QA Model
    # Type: Generative (creates new text, not just extracts)
    # Expected: Medium accuracy, slower speed, no confidence scores
    # Use case: When you need creative/paraphrased answers
    "consciousAI/question-answering-generative-t5-v1-base-s-q-c",
    
    # 2. RoBERTa-based Extractive QA Model
    # Type: Extractive (finds answer in context)
    # Expected: High accuracy, fast speed, good confidence scores
    # Use case: Best balance - handles unanswerable questions well
    # Trained on: SQuAD 2.0 (includes unanswerable questions)
    "deepset/roberta-base-squad2",
    
    # 3. BERT Large Extractive QA Model
    # Type: Extractive (large model, high accuracy)
    # Expected: Very high accuracy, slower speed, good confidence scores
    # Use case: When accuracy is more important than speed
    # Size: Large (~340M parameters) - needs more memory
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",
    
    # 4. RAG-Optimized Model
    # Type: RAG-specific (designed for retrieval-augmented generation)
    # Expected: Good for RAG systems, medium speed
    # Use case: Specifically designed for RAG workflows
    # Size: 8B parameters (very large, needs GPU)
    "gasolsun/DynamicRAG-8B",
    
    # ========================================================================
    # ADDITIONAL MODELS (2) - For comparison and learning
    # ========================================================================
    
    # 5. DistilBERT (Smaller, Faster)
    # Type: Extractive (distilled from BERT)
    # Expected: Good accuracy, very fast speed, smaller size
    # Use case: Speed vs accuracy trade-off - 60% smaller, 60% faster than BERT
    # Trade-off: Slightly less accurate than full BERT but much faster
    "distilbert-base-uncased-distilled-squad",
    
    # 6. BERT Tiny (Minimal Size)
    # Type: Extractive (tiny version)
    # Expected: Lower accuracy, very fast speed, tiny size
    # Use case: Testing on resource-constrained devices
    # Trade-off: Fastest but least accurate
    "mrm8488/bert-tiny-finetuned-squadv2",
    
    # Note: Mistral-7B-Instruct-v0.3 is NOT included here because:
    # - Mistral is used ONLY for generating Q&A pairs (Step 2) - see MISTRAL_MODEL_ID
    # - It's NOT meant to be evaluated as a QA model
    # - The workflow: Mistral generates Q&A → embeddings created → QA models evaluate
    # - Mistral is a text generation model, not optimized for QA tasks
]

SIMILARITY_THRESHOLD = 0.7  # Threshold for determining if question is answerable


# ============================================================================
# DEVICE DETECTION & CONFIGURATION (Reused from prompt_engineering_basics.py)
# ============================================================================

def check_gpu_availability() -> Tuple[bool, str, str]:
    """Check if GPU is available."""
    try:
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0), torch.version.cuda
    except Exception:
        pass
    return False, None, None


def create_gpu_config(gpu_name: str, cuda_version: str) -> Dict[str, Any]:
    """Create GPU configuration dictionary."""
    return {
        "device": "cuda",
        "device_name": "cuda",
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "is_cpu": False,
        "is_gpu": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "max_new_tokens": 512,
        "device_map": "auto",
        "pipeline_device": 0,
    }


def create_cpu_config() -> Dict[str, Any]:
    """Create CPU configuration dictionary."""
    return {
        "device": "cpu",
        "device_name": "cpu",
        "gpu_name": None,
        "cuda_version": None,
        "is_cpu": True,
        "is_gpu": False,
        "torch_dtype": torch.float32,
        "max_new_tokens": 256,
        "device_map": None,
        "pipeline_device": -1,
    }


def get_device_configuration() -> Dict[str, Any]:
    """Check device availability and return complete configuration."""
    print("STEP 1: Checking Your System")
    print("=" * 60)
    print("\n🔍 Detecting device (CPU/GPU)...")
    
    is_gpu, gpu_name, cuda_version = check_gpu_availability()
    
    if is_gpu:
        print(f"   ✅ GPU Available: {gpu_name}")
        print(f"   ✅ CUDA Version: {cuda_version}")
        return create_gpu_config(gpu_name, cuda_version)
    else:
        print("   ⚠️  GPU NOT detected - using CPU")
        print("   💡 CPU works fine, but GPU is much faster!")
        return create_cpu_config()


# ============================================================================
# AUTHENTICATION
# ============================================================================

def get_hf_token() -> str:
    """Get Hugging Face token from environment."""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("\n❌ Hugging Face token not found!")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        print("   Create .env file: HUGGINGFACE_HUB_TOKEN=your_token_here")
        sys.exit(1)
    return token


def setup_token() -> str:
    """Setup and validate Hugging Face token."""
    token = get_hf_token()
    print("✅ Hugging Face token loaded successfully!")
    preview = f"{token[:10]}...{token[-4:]}" if len(token) > 14 else "****"
    print(f"   Token preview: {preview}\n")
    return token


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_mistral_model(model_id: str, hf_token: str, device_config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Load Mistral model and tokenizer."""
    print("=" * 60)
    print("STEP 2: Loading Mistral Model")
    print("=" * 60)
    print(f"\n📚 Loading: {model_id}")
    print(f"⏳ This may take 1-2 minutes on first run (downloading model)...")
    print(f"💡 Note: Verbose warnings are suppressed for cleaner output\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    if device_config["is_cpu"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            dtype=device_config["torch_dtype"],
            low_cpu_mem_usage=True,
        ).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            dtype=device_config["torch_dtype"],
            device_map=device_config["device_map"],
        )
    
    print("✅ Mistral model loaded successfully!\n")
    return model, tokenizer


def create_mistral_pipeline(model: Any, tokenizer: Any, device_config: Dict[str, Any]) -> Any:
    """Create text generation pipeline from Mistral model."""
    print("Setting up Mistral pipeline...")
    
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": device_config["max_new_tokens"],
        "do_sample": True,
        "num_return_sequences": 1,
    }
    
    if device_config["device_map"] is not None:
        kwargs["device_map"] = device_config["device_map"]
    else:
        kwargs["device"] = device_config["pipeline_device"]
    
    chatbot = pipeline("text-generation", **kwargs)
    print("✅ Pipeline ready!\n")
    return chatbot


# ============================================================================
# STEP 1: CREATE SYSTEM PROMPT
# ============================================================================

def create_system_prompt(business_name: str, role: str) -> str:
    """
    Create a system prompt for the business context.
    
    Args:
        business_name: Name of the business/organization
        role: Professional role for the AI (e.g., "AI Solutions Consultant")
    
    Returns:
        System prompt string
    """
    print("=" * 60)
    print("STEP 1: Creating System Prompt")
    print("=" * 60)
    print(f"\n📋 Business: {business_name}")
    print(f"📋 Role: {role}\n")
    
    system_prompt = f"""You are a {role} at {business_name}. 
You are knowledgeable, professional, and helpful. 
You provide accurate information about {business_name}'s services, pricing, processes, and expertise.
Always be courteous and aim to help customers understand how {business_name} can assist them."""
    
    print("✅ System prompt created:")
    print(f"   {system_prompt}\n")
    return system_prompt


# ============================================================================
# STEP 2: GENERATE Q&A DATABASE
# ============================================================================

def parse_qa_json(response_text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Parse Q&A pairs from JSON response.
    
    KISS Principle: Simple JSON parsing instead of complex text parsing.
    
    Returns:
        Tuple of (answerable_pairs, unanswerable_pairs)
    """
    try:
        # Try to extract JSON from response (might have markdown code blocks)
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Extract answerable and unanswerable pairs
        answerable = data.get("answerable", [])
        unanswerable = data.get("unanswerable", [])
        
        # Fallback: if old format (just "qa_pairs"), treat all as answerable
        if not answerable and not unanswerable and "qa_pairs" in data:
            answerable = data["qa_pairs"]
            unanswerable = []
        
        return answerable, unanswerable
    except json.JSONDecodeError:
        # Fallback: return empty lists if JSON parsing fails
        return [], []


def generate_qa_database(
    chatbot: Any, 
    system_prompt: str, 
    business_name: str, 
    max_retries: int = 2
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Generate Q&A database using Mistral.
    
    Args:
        chatbot: Mistral pipeline
        system_prompt: System prompt for business context
        business_name: Name of the business
        max_retries: Maximum number of retries if not enough pairs generated
    
    Returns:
        Tuple of (answerable_pairs, unanswerable_pairs)
    """
    # Generate new Q&A pairs
    print("=" * 60)
    print("STEP 2: Generating Q&A Database")
    print("=" * 60)
    print("\n📚 Generating 15 question pairs: 7 answerable + 7 unanswerable")
    print("   Answerable: Questions the business CAN answer (knowledge base)")
    print("   Unanswerable: Questions the business CANNOT answer (outside expertise)")
    print("   This may take 30-60 seconds...\n")
    
    # KISS Principle: Request JSON format directly - no parsing needed!
    prompt = f"""Generate 15 question-answer pairs for {business_name}:

REQUIREMENTS:
- Generate exactly 15 pairs total
- 7 answerable pairs: Questions about {business_name}'s services, pricing, processes, technical details, contact info (these go in knowledge base)
- 7 unanswerable pairs: Questions about competitor info, unrelated topics, personal details, things outside {business_name}'s expertise (these are for testing)
- 1 additional pair (your choice: answerable or unanswerable)

Return ONLY valid JSON format (no other text):

JSON FORMAT:
{{
  "answerable": [
    {{"question": "What services do you offer?", "answer": "We offer..."}},
    {{"question": "How much does it cost?", "answer": "Our pricing..."}}
  ],
  "unanswerable": [
    {{"question": "What do your competitors charge?", "answer": "I don't have information about competitors."}},
    {{"question": "What's the weather today?", "answer": "I cannot provide weather information."}}
  ]
}}

Return ONLY the JSON, nothing else."""

    answerable_pairs = []
    unanswerable_pairs = []
    attempts = 0
    
    while (len(answerable_pairs) < 7 or len(unanswerable_pairs) < 7) and attempts <= max_retries:
        attempts += 1
        if attempts > 1:
            print(f"   🔄 Retry attempt {attempts-1}/{max_retries} (need at least 10 pairs)...\n")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        start_time = time.time()
        with torch.no_grad():
            # Use longer generation for Q&A pairs (override pipeline's max_new_tokens)
            result = chatbot(
                messages,
                max_new_tokens=1024,  # More tokens for longer output (10-15 Q&A pairs)
                do_sample=True,
                top_k=10
            )
        generation_time = time.time() - start_time
        
        # Extract response
        response_text = result[0]["generated_text"][-1]["content"]
        
        # Parse Q&A pairs from JSON (KISS: simple JSON parsing)
        answerable, unanswerable = parse_qa_json(response_text)
        
        answerable_pairs = answerable
        unanswerable_pairs = unanswerable
        
        total_pairs = len(answerable_pairs) + len(unanswerable_pairs)
        print(f"   Attempt {attempts}: Generated {len(answerable_pairs)} answerable + {len(unanswerable_pairs)} unanswerable = {total_pairs} total pairs in {generation_time:.2f} seconds")
        
        if len(answerable_pairs) >= 7 and len(unanswerable_pairs) >= 7:
            break
    
    print(f"\n✅ Final result:")
    print(f"   Answerable pairs: {len(answerable_pairs)} (target: 7+)")
    print(f"   Unanswerable pairs: {len(unanswerable_pairs)} (target: 7+)")
    print(f"   Total: {len(answerable_pairs) + len(unanswerable_pairs)} pairs")
    
    print("\n📋 Sample answerable pairs (knowledge base):")
    for i, qa in enumerate(answerable_pairs[:3], 1):
        print(f"\n   {i}. Q: {qa['question'][:70]}...")
        print(f"      A: {qa['answer'][:70]}...")
    
    print("\n📋 Sample unanswerable pairs (for testing):")
    for i, qa in enumerate(unanswerable_pairs[:3], 1):
        print(f"\n   {i}. Q: {qa['question'][:70]}...")
        print(f"      A: {qa['answer'][:70]}...")
    
    if len(answerable_pairs) < 7 or len(unanswerable_pairs) < 7:
        print(f"\n⚠️  Warning: Need at least 7 of each type")
        print(f"   Got: {len(answerable_pairs)} answerable, {len(unanswerable_pairs)} unanswerable")
        print("   The script will continue, but you may want to regenerate.")
    else:
        print(f"\n✅ Excellent! Generated {len(answerable_pairs)} answerable + {len(unanswerable_pairs)} unanswerable pairs")
    
    print()
    return answerable_pairs, unanswerable_pairs


# ============================================================================
# STEP 3: IMPLEMENT FAISS VECTOR DATABASE
# ============================================================================

def create_embeddings(questions: List[str], embedding_model: Any) -> np.ndarray:
    """Create embeddings for questions using sentence transformers."""
    print("Creating embeddings for questions...")
    # Suppress progress bar for cleaner output (students can see final results)
    embeddings = embedding_model.encode(questions, show_progress_bar=False)
    return embeddings.astype('float32')


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embeddings)
    return index


def search_similar_questions(
    query: str,
    embedding_model: Any,
    faiss_index: faiss.Index,
    qa_database: List[Dict[str, str]],
    top_k: int = 3
) -> List[Tuple[Dict[str, str], float]]:
    """
    Search for similar questions in the database.
    
    Returns:
        List of (qa_pair, similarity_score) tuples
    """
    # Create embedding for query
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding.astype('float32')
    
    # Search in FAISS index
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(qa_database):
            # Convert distance to similarity (lower distance = higher similarity)
            similarity = 1.0 / (1.0 + dist)
            results.append((qa_database[idx], similarity))
    
    return results


def implement_faiss_database(qa_database: List[Dict[str, str]], hf_token: str) -> Tuple[Any, faiss.Index]:
    """
    Implement FAISS vector database.
    
    Args:
        qa_database: List of Q&A pairs
        hf_token: Hugging Face token
    
    Returns:
        Tuple of (embedding_model, faiss_index)
    """
    print("=" * 60)
    print("STEP 3: Implementing FAISS Vector Database")
    print("=" * 60)
    print(f"\n📚 Loading embedding model: {EMBEDDING_MODEL_ID}")
    print("   This converts text to numerical vectors for similarity search...\n")
    
    # Load embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)
    print("✅ Embedding model loaded!\n")
    
    # Extract questions
    questions = [qa["question"] for qa in qa_database]
    print(f"📋 Creating embeddings for {len(questions)} questions...")
    
    # Create embeddings
    embeddings = create_embeddings(questions, embedding_model)
    print(f"✅ Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}\n")
    
    # Build FAISS index
    print("📋 Building FAISS index...")
    faiss_index = build_faiss_index(embeddings)
    print(f"✅ FAISS index built with {faiss_index.ntotal} vectors\n")
    
    # Test search
    print("🧪 Testing search functionality...")
    test_query = questions[0] if questions else "What services do you offer?"
    results = search_similar_questions(test_query, embedding_model, faiss_index, qa_database, top_k=3)
    
    print(f"   Query: {test_query[:50]}...")
    print("   Top matches:")
    for i, (qa, similarity) in enumerate(results, 1):
        print(f"   {i}. Similarity: {similarity:.3f} - {qa['question'][:50]}...")
    print()
    
    return embedding_model, faiss_index


# ============================================================================
# STEP 4: CREATE TEST QUESTIONS
# ============================================================================

def generate_test_questions(
    chatbot: Any,
    system_prompt: str,
    question_type: str,
    business_name: str
) -> List[str]:
    """
    Generate test questions using Mistral.
    
    Args:
        question_type: "answerable" or "unanswerable"
    """
    if question_type == "answerable":
        prompt = f"""Generate 5 questions that {business_name} CAN answer about their services, pricing, processes, or expertise.
Make them realistic customer questions."""
    else:
        prompt = f"""Generate 5 questions that {business_name} CANNOT answer.
These should be about:
- Competitor information
- Unrelated topics outside their expertise
- Personal details
- Information not in their knowledge base"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    with torch.no_grad():
        result = chatbot(messages)
    
    response_text = result[0]["generated_text"][-1]["content"]
    
    # Parse questions (numbered list or Q: format)
    questions = []
    for line in response_text.split('\n'):
        line = line.strip()
        # Remove numbering (1., 2., etc.) or Q: prefix
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^Q[\.:]\s*', '', line, flags=re.IGNORECASE)
        if line and len(line) > 10:  # Valid question
            questions.append(line)
    
    return questions[:5]  # Return up to 5 questions


def create_test_questions(chatbot: Any, system_prompt: str, business_name: str) -> Tuple[List[str], List[str]]:
    """Create both answerable and unanswerable test questions."""
    print("=" * 60)
    print("STEP 4: Creating Test Questions")
    print("=" * 60)
    print("\n📚 Generating answerable questions...")
    answerable = generate_test_questions(chatbot, system_prompt, "answerable", business_name)
    print(f"✅ Generated {len(answerable)} answerable questions\n")
    
    print("📚 Generating unanswerable questions...")
    unanswerable = generate_test_questions(chatbot, system_prompt, "unanswerable", business_name)
    print(f"✅ Generated {len(unanswerable)} unanswerable questions\n")
    
    print("📋 Sample questions:")
    print("\n   Answerable:")
    for i, q in enumerate(answerable[:3], 1):
        print(f"   {i}. {q[:60]}...")
    print("\n   Unanswerable:")
    for i, q in enumerate(unanswerable[:3], 1):
        print(f"   {i}. {q[:60]}...")
    print()
    
    return answerable, unanswerable


# ============================================================================
# STEP 5: TEST RAG SYSTEM
# ============================================================================

def test_rag_system(
    questions: List[str],
    embedding_model: Any,
    faiss_index: faiss.Index,
    qa_database: List[Dict[str, str]],
    threshold: float = SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """Test RAG system with questions and return performance metrics."""
    results = []
    correct = 0
    
    for question in questions:
        search_results = search_similar_questions(
            question, embedding_model, faiss_index, qa_database, top_k=1
        )
        
        if search_results:
            best_match, similarity = search_results[0]
            is_answerable = similarity >= threshold
            results.append({
                "question": question,
                "similarity": similarity,
                "is_answerable": is_answerable,
                "matched_qa": best_match
            })
            if is_answerable:
                correct += 1
        else:
            results.append({
                "question": question,
                "similarity": 0.0,
                "is_answerable": False,
                "matched_qa": None
            })
    
    accuracy = correct / len(questions) if questions else 0.0
    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0.0
    
    return {
        "results": results,
        "accuracy": accuracy,
        "avg_similarity": avg_similarity,
        "correct": correct,
        "total": len(questions)
    }


def implement_and_test_questions(
    answerable: List[str],
    unanswerable: List[str],
    embedding_model: Any,
    faiss_index: faiss.Index,
    qa_database: List[Dict[str, str]]
) -> None:
    """Test RAG system with both question types."""
    print("=" * 60)
    print("STEP 5: Testing RAG System")
    print("=" * 60)
    print(f"\n📊 Similarity threshold: {SIMILARITY_THRESHOLD}")
    print("   (Questions with similarity >= threshold are considered answerable)\n")
    
    # Test answerable questions
    print("🧪 Testing answerable questions...")
    answerable_results = test_rag_system(
        answerable, embedding_model, faiss_index, qa_database
    )
    
    print(f"   ✅ Accuracy: {answerable_results['accuracy']:.1%} ({answerable_results['correct']}/{answerable_results['total']})")
    print(f"   📊 Average similarity: {answerable_results['avg_similarity']:.3f}\n")
    
    # Test unanswerable questions
    print("🧪 Testing unanswerable questions...")
    unanswerable_results = test_rag_system(
        unanswerable, embedding_model, faiss_index, qa_database
    )
    
    # For unanswerable, we want LOW similarity (so accuracy = 1 - correct/total)
    unanswerable_correct = unanswerable_results['total'] - unanswerable_results['correct']
    unanswerable_accuracy = unanswerable_correct / unanswerable_results['total'] if unanswerable_results['total'] > 0 else 0.0
    
    print(f"   ✅ Accuracy: {unanswerable_accuracy:.1%} ({unanswerable_correct}/{unanswerable_results['total']} correctly identified as unanswerable)")
    print(f"   📊 Average similarity: {unanswerable_results['avg_similarity']:.3f}\n")
    
    # Overall performance
    total_correct = answerable_results['correct'] + unanswerable_correct
    total_questions = answerable_results['total'] + unanswerable_results['total']
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    
    print("=" * 60)
    print("📊 Overall Performance Summary")
    print("=" * 60)
    print(f"   Overall Accuracy: {overall_accuracy:.1%} ({total_correct}/{total_questions})")
    print(f"   Answerable Questions: {answerable_results['accuracy']:.1%} accuracy")
    print(f"   Unanswerable Questions: {unanswerable_accuracy:.1%} accuracy")
    print("=" * 60)
    print()


# ============================================================================
# STEP 6: MODEL EVALUATION AND RANKING
# ============================================================================

def evaluate_qa_model(
    model_id: str,
    question: str,
    context: str,
    hf_token: str
) -> Dict[str, Any]:
    """
    Evaluate a single QA model on a question-context pair.
    
    Why Explicit QA Models Help:
    - They return confidence scores (0.0 to 1.0) showing how sure they are
    - They're optimized for speed (milliseconds vs seconds)
    - They handle edge cases better (unanswerable questions, ambiguous contexts)
    - They provide structured output (answer, score, start/end positions)
    
    General models (like Qwen) require:
    - Manual prompt engineering
    - No confidence scores
    - Slower inference
    - Less reliable for QA tasks
    """
    try:
        # Suppress warnings during model loading and inference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Try QA pipeline first (for explicit QA models)
            start_time = time.time()
            try:
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model_id,
                    token=hf_token
                )
                load_time = time.time() - start_time
                
                start_time = time.time()
                result = qa_pipeline(question=question, context=context)
                inference_time = time.time() - start_time
                
                return {
                    "model_id": model_id,
                    "answer": result.get("answer", ""),
                    "score": result.get("score", 0.0),  # Confidence score (QA models provide this!)
                    "load_time": load_time,
                    "inference_time": inference_time,
                    "success": True,
                    "model_type": "explicit_qa"  # Explicit QA model
                }
            except (ValueError, OSError, RuntimeError) as e:
                # Fallback: Try as text generation model (for general models like Qwen/Mistral)
                # This shows why explicit QA models are better - they work directly!
                try:
                    gen_pipeline = pipeline(
                        "text-generation",
                        model=model_id,
                        token=hf_token,
                        max_new_tokens=50,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    load_time = time.time() - start_time
                    
                    # Create prompt for general model
                    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                    start_time = time.time()
                    result = gen_pipeline(prompt, return_full_text=False)
                    inference_time = time.time() - start_time
                    
                    answer = result[0]["generated_text"].strip()
                    
                    return {
                        "model_id": model_id,
                        "answer": answer,
                        "score": 0.5,  # No confidence score available (general model limitation)
                        "load_time": load_time,
                        "inference_time": inference_time,
                        "success": True,
                        "model_type": "general"  # General model (not QA-specific)
                    }
                except Exception as gen_error:
                    # Text generation also failed - likely model too large for CPU
                    raise Exception(f"Text generation failed: {str(gen_error)}. Model may be too large for CPU. Try GPU or use a smaller model.")
    except Exception as e:
        # Extract error message (truncate if too long)
        error_msg = str(e)
        if len(error_msg) > 150:
            error_msg = error_msg[:147] + "..."
        return {
            "model_id": model_id,
            "answer": "",
            "score": 0.0,
            "load_time": 0.0,
            "inference_time": 0.0,
            "success": False,
            "error": error_msg,
            "model_type": "unknown"
        }


def rank_qa_models(
    qa_database: List[Dict[str, str]],
    embedding_model: Any,
    faiss_index: faiss.Index,
    hf_token: str,
    test_questions: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate and rank QA models.
    
    Evaluation Tasks (from class exercise):
    1. Speed: Measure response time (milliseconds)
    2. Confidence Scores: How sure is the model? (0.0 to 1.0)
    3. Answer Quality: How correct/helpful is the answer?
    
    Models are tested on questions from the Q&A database.
    """
    print("=" * 60)
    print("STEP 6: Model Experimentation and Ranking")
    print("=" * 60)
    print(f"\n📚 Testing {len(QA_MODELS)} QA models...")
    print("\n📊 Evaluation Tasks:")
    print("   1. Speed: Response time (milliseconds)")
    print("   2. Confidence Scores: Model certainty (0.0 to 1.0)")
    print("   3. Answer Quality: Correctness and helpfulness")
    print("\n💡 What You'll See:")
    print("   - Each model will be tested on sample questions")
    print("   - Results show confidence score and speed")
    print("   - Models are ranked by performance")
    print("   - Note: Model loading may take time (first time only)")
    print("\n   ⏳ This may take several minutes...\n")
    
    # Use test questions if provided, otherwise use Q&A database
    if test_questions is None:
        test_questions = [qa["question"] for qa in qa_database[:3]]  # Test on 3 questions
    
    # Build context from answers
    context = " ".join([qa["answer"] for qa in qa_database[:3]])
    
    model_results = []
    
    for i, model_id in enumerate(QA_MODELS, 1):
        # Show progress with clear formatting
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        print(f"   [{i}/{len(QA_MODELS)}] Testing {model_name}...", end=" ", flush=True)
        
        # Test on a single question for evaluation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            question = test_questions[0]  # Test on first question
            result = evaluate_qa_model(model_id, question, context, hf_token)
        
        model_results.append(result)
        
        # Show result on same line
        if result["success"]:
            print(f"✅ Score: {result['score']:.3f} | Time: {result['inference_time']:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            # Show helpful error message for students
            if "too large" in error_msg.lower() or "out of memory" in error_msg.lower():
                print(f"❌ Failed: Model too large for CPU (needs GPU or smaller model)")
            else:
                # Truncate long error messages
                short_error = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
                print(f"❌ Failed: {short_error}")
        print()
    
    # Rank models by confidence score (primary) and speed (secondary)
    model_results.sort(key=lambda x: (x.get("score", 0.0), -x.get("inference_time", float('inf'))), reverse=True)
    
    # Print ranking as a table
    print("=" * 60)
    print("📊 Model Ranking - Review and Choose the Best Model for Your Use Case")
    print("=" * 60)
    print()
    print(f"{'Rank':<6} {'Model':<50} {'Type':<15} {'Confidence':<12} {'Speed (s)':<12}")
    print("-" * 100)
    
    for i, result in enumerate(model_results, 1):
        model_name = result['model_id'].split('/')[-1] if '/' in result['model_id'] else result['model_id']
        if len(model_name) > 48:
            model_name = model_name[:45] + "..."
        
        if result["success"]:
            model_type = result.get("model_type", "unknown")
            type_label = "QA-Specific" if model_type == "explicit_qa" else "General"
            confidence = f"{result['score']:.3f}"
            speed = f"{result['inference_time']:.3f}"
        else:
            type_label = "Failed"
            confidence = "N/A"
            speed = "N/A"
        
        print(f"{i:<6} {model_name:<50} {type_label:<15} {confidence:<12} {speed:<12}")
    
    print("-" * 100)
    print("\n💡 Understanding the Results:")
    print("   - Confidence Score: How sure the model is (0.0 = unsure, 1.0 = very sure)")
    print("   - Speed: Lower is better (response time in seconds)")
    print("   - Model Type: QA-Specific models are optimized for question-answering")
    print("=" * 60)
    print()
    
    return model_results


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main() -> None:
    """Main orchestration function."""
    print("=" * 60)
    print("RAG System Exercise - Building a Complete RAG System")
    print("=" * 60)
    print()
    
    # Configuration
    device_config = get_device_configuration()
    hf_token = setup_token()
    
    # Business configuration (customize these)
    BUSINESS_NAME = "TechStart Solutions"
    ROLE = "AI Solutions Consultant"
    
    try:
        # Load Mistral model
        model, tokenizer = load_mistral_model(MISTRAL_MODEL_ID, hf_token, device_config)
        chatbot = create_mistral_pipeline(model, tokenizer, device_config)
        
        # Step 1: Create system prompt
        system_prompt = create_system_prompt(BUSINESS_NAME, ROLE)
        
        # Step 2: Generate Q&A database (answerable for knowledge base, unanswerable for testing)
        answerable_qa, unanswerable_qa = generate_qa_database(
            chatbot, 
            system_prompt, 
            BUSINESS_NAME
        )
        
        # Use answerable pairs as the knowledge base
        qa_database = answerable_qa
        
        # Step 3: Implement FAISS database (using answerable pairs as knowledge base)
        embedding_model, faiss_index = implement_faiss_database(qa_database, hf_token)
        
        # Step 4: Use unanswerable pairs from Step 2 as test questions
        # Also generate a few more answerable test questions
        print("=" * 60)
        print("STEP 4: Preparing Test Questions")
        print("=" * 60)
        print("\n📚 Using unanswerable pairs from Step 2 as test questions")
        print(f"   Unanswerable test questions: {len(unanswerable_qa)}")
        
        # Generate a few more answerable test questions for better testing
        print("\n📚 Generating additional answerable test questions...")
        additional_answerable = generate_test_questions(chatbot, system_prompt, "answerable", BUSINESS_NAME)
        
        # Combine: use generated answerable + additional answerable as test set
        # Extract questions from dicts and combine with additional answerable questions (strings)
        answerable = [qa["question"] for qa in answerable_qa[:5]] + additional_answerable[:2]  # 5 from knowledge base + 2 new = 7 total
        unanswerable = [qa["question"] for qa in unanswerable_qa[:7]]  # Use 7 unanswerable from Step 2
        
        print(f"   Answerable test questions: {len(answerable)}")
        print(f"   Unanswerable test questions: {len(unanswerable)}")
        print()
        
        # Step 5: Test RAG system
        implement_and_test_questions(
            answerable, unanswerable, embedding_model, faiss_index, qa_database
        )
        
        # Step 6: Rank QA models (test on answerable questions)
        test_questions = [qa["question"] for qa in answerable_qa[:5]]  # Use 5 questions for testing
        model_rankings = rank_qa_models(
            qa_database, 
            embedding_model, 
            faiss_index, 
            hf_token,
            test_questions=test_questions
        )
        
        # Final summary
        print("=" * 60)
        print("✅ RAG System Exercise Completed!")
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

