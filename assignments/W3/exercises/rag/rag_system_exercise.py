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
from typing import Dict, Tuple, List, Any, Optional

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Optional dependency

# Import required libraries
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import torch
    import numpy as np
    import faiss
except ImportError:
    print("❌ Required packages not installed!")
    print("   Install with: pip install transformers torch sentence-transformers faiss-cpu")
    sys.exit(1)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight and fast

QA_MODELS = [
    "consciousAI/question-answering-generative-t5-v1-base-s-q-c",
    "deepset/roberta-base-squad2",
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",
    "gasolsun/DynamicRAG-8B",
    # Add 2 more models of your choice
    "distilbert-base-uncased-distilled-squad",
    "mrm8488/bert-tiny-finetuned-squadv2",
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
    print(f"⏳ This may take 1-2 minutes on first run (downloading model)...\n")
    
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

def parse_qa_pairs(text: str) -> List[Dict[str, str]]:
    """
    Parse Q&A pairs from generated text.
    
    Expected format:
    Q: question text
    A: answer text
    """
    qa_pairs = []
    lines = text.split('\n')
    
    current_q = None
    current_a = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q:') or line.startswith('Q.'):
            if current_q and current_a:
                qa_pairs.append({"question": current_q, "answer": current_a})
            current_q = line[2:].strip()
            current_a = None
        elif line.startswith('A:') or line.startswith('A.'):
            current_a = line[2:].strip()
        elif current_a is not None:
            current_a += " " + line
        elif current_q is not None and not line.startswith('Q'):
            current_q += " " + line
    
    # Add last pair
    if current_q and current_a:
        qa_pairs.append({"question": current_q, "answer": current_a})
    
    return qa_pairs


def generate_qa_database(chatbot: Any, system_prompt: str, business_name: str) -> List[Dict[str, str]]:
    """
    Generate Q&A database using Mistral.
    
    Args:
        chatbot: Mistral pipeline
        system_prompt: System prompt for business context
        business_name: Name of the business
    
    Returns:
        List of Q&A dictionaries
    """
    print("=" * 60)
    print("STEP 2: Generating Q&A Database")
    print("=" * 60)
    print("\n📚 Asking Mistral to generate 10-15 Q&A pairs...")
    print("   This may take 30-60 seconds...\n")
    
    prompt = f"""Generate 10-15 realistic question-answer pairs that customers might ask about {business_name}.

Cover these topics:
- Services offered
- Pricing information
- Processes and procedures
- Technical details
- Contact information

Format each pair as:
Q: [question]
A: [answer]

Make the questions realistic and the answers helpful and informative."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    start_time = time.time()
    with torch.no_grad():
        result = chatbot(messages)
    generation_time = time.time() - start_time
    
    # Extract response
    response_text = result[0]["generated_text"][-1]["content"]
    
    # Parse Q&A pairs
    qa_database = parse_qa_pairs(response_text)
    
    print(f"✅ Generated {len(qa_database)} Q&A pairs in {generation_time:.2f} seconds")
    print("\n📋 Sample Q&A pairs:")
    for i, qa in enumerate(qa_database[:3], 1):
        print(f"\n   {i}. Q: {qa['question'][:60]}...")
        print(f"      A: {qa['answer'][:60]}...")
    
    if len(qa_database) < 10:
        print(f"\n⚠️  Only {len(qa_database)} pairs generated. You may want to regenerate.")
    
    print()
    return qa_database


# ============================================================================
# STEP 3: IMPLEMENT FAISS VECTOR DATABASE
# ============================================================================

def create_embeddings(questions: List[str], embedding_model: Any) -> np.ndarray:
    """Create embeddings for questions using sentence transformers."""
    print("Creating embeddings for questions...")
    embeddings = embedding_model.encode(questions, show_progress_bar=True)
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
    """Evaluate a single QA model on a question-context pair."""
    try:
        start_time = time.time()
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
            "score": result.get("score", 0.0),
            "load_time": load_time,
            "inference_time": inference_time,
            "success": True
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "answer": "",
            "score": 0.0,
            "load_time": 0.0,
            "inference_time": 0.0,
            "success": False,
            "error": str(e)
        }


def rank_qa_models(
    qa_database: List[Dict[str, str]],
    embedding_model: Any,
    faiss_index: faiss.Index,
    hf_token: str
) -> List[Dict[str, Any]]:
    """Evaluate and rank QA models."""
    print("=" * 60)
    print("STEP 6: Model Experimentation and Ranking")
    print("=" * 60)
    print(f"\n📚 Testing {len(QA_MODELS)} QA models...")
    print("   This may take several minutes...\n")
    
    # Select a sample question and its context
    sample_qa = qa_database[0] if qa_database else {"question": "What services do you offer?", "answer": "We offer various services."}
    sample_question = sample_qa["question"]
    sample_context = sample_qa["answer"] + " " + " ".join([qa["answer"] for qa in qa_database[1:3]])
    
    model_results = []
    
    for i, model_id in enumerate(QA_MODELS, 1):
        print(f"   [{i}/{len(QA_MODELS)}] Testing {model_id}...")
        result = evaluate_qa_model(model_id, sample_question, sample_context, hf_token)
        model_results.append(result)
        
        if result["success"]:
            print(f"      ✅ Score: {result['score']:.3f}, Time: {result['inference_time']:.2f}s")
        else:
            print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
        print()
    
    # Rank models by composite score (score * 0.6 + speed_score * 0.4)
    for result in model_results:
        if result["success"] and result["inference_time"] > 0:
            # Normalize speed (faster = higher score, max 1.0)
            max_time = max(r["inference_time"] for r in model_results if r["success"] and r["inference_time"] > 0)
            speed_score = 1.0 - (result["inference_time"] / max_time) if max_time > 0 else 0.0
            result["composite_score"] = result["score"] * 0.6 + speed_score * 0.4
        else:
            result["composite_score"] = 0.0
    
    # Sort by composite score
    model_results.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Print ranking
    print("=" * 60)
    print("📊 Model Ranking (Best to Worst)")
    print("=" * 60)
    for i, result in enumerate(model_results, 1):
        print(f"\n   {i}. {result['model_id']}")
        if result["success"]:
            print(f"      Confidence Score: {result['score']:.3f}")
            print(f"      Inference Time: {result['inference_time']:.2f}s")
            print(f"      Composite Score: {result['composite_score']:.3f}")
        else:
            print(f"      ❌ Failed to load/run")
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
        
        # Step 2: Generate Q&A database
        qa_database = generate_qa_database(chatbot, system_prompt, BUSINESS_NAME)
        
        if len(qa_database) < 5:
            print("⚠️  Warning: Not enough Q&A pairs generated. Continuing anyway...\n")
        
        # Step 3: Implement FAISS database
        embedding_model, faiss_index = implement_faiss_database(qa_database, hf_token)
        
        # Step 4: Create test questions
        answerable, unanswerable = create_test_questions(chatbot, system_prompt, BUSINESS_NAME)
        
        # Step 5: Test RAG system
        implement_and_test_questions(
            answerable, unanswerable, embedding_model, faiss_index, qa_database
        )
        
        # Step 6: Rank QA models
        model_rankings = rank_qa_models(qa_database, embedding_model, faiss_index, hf_token)
        
        # Final summary
        print("=" * 60)
        print("✅ RAG System Exercise Completed!")
        print("=" * 60)
        print("\n💡 Next Steps:")
        print("   1. Review the model rankings and choose the best model for your use case")
        print("   2. Adjust the similarity threshold if needed")
        print("   3. Expand your Q&A database with more examples")
        print("   4. Test with real user questions")
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

