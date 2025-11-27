#!/usr/bin/env python3
"""
Prompt Engineering Basics - Standalone Python Script
Extracted from W3__Prompt_Engineering_Basics.ipynb

This script demonstrates:
- Basic prompt engineering with Qwen3-0.6B or Mistral-7B-Instruct
- Model interaction using Hugging Face Transformers
- Pipeline vs Direct Model Loading (using same model instance)
- Device optimization (CPU/GPU)

Performance Notes:
- CPU inference: ~5-8 seconds per response (optimized with torch.no_grad())
- GPU inference: ~1-2 seconds per response (10-50x faster)
- Model loading: 1-2 minutes on CPU, 30-60 seconds on GPU
- Optimizations applied: torch.no_grad(), greedy decoding on CPU, reduced max tokens
"""

import sys
import os
import time
import json
from typing import Dict, Tuple, List, Any

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Optional dependency

# Import required libraries
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print("❌ Required packages not installed!")
    print("   Install with: pip install transformers torch sentence-transformers")
    sys.exit(1)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

AVAILABLE_MODELS = {
    "qwen": {
        "model_id": "Qwen/Qwen3-0.6B",
        "name": "Qwen3-0.6B",
        "description": "Latest Qwen3 series, lightweight (~1.2GB disk, ~2.5GB RAM)",
        "cpu_ram_required": 2.5,
        "gpu_vram_required": 1.5,
        "recommended_for": "Low RAM systems, CPU inference",
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral-7B-Instruct-v0.3",
        "description": "High-quality 7B parameter model (~14GB RAM on CPU, ~14GB VRAM on GPU)",
        "cpu_ram_required": 14.0,
        "gpu_vram_required": 14.0,
        "recommended_for": "GPU systems, high-quality responses",
    },
}

DEFAULT_MODEL = "mistral"


# ============================================================================
# DEVICE DETECTION & CONFIGURATION
# ============================================================================

def check_gpu_availability() -> Tuple[bool, str, str]:
    """
    Check if GPU is available.
    
    Returns:
        Tuple of (is_available, gpu_name, cuda_version)
    """
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
        "do_sample": True,
        "top_k": 10,
        "load_time_estimate": "30-60 seconds",
        "generation_time_estimate": "1-2 seconds per response",
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
        "do_sample": False,
        "top_k": None,
        "load_time_estimate": "1-2 minutes",
        "generation_time_estimate": "3-8 seconds per response",
    }


def get_device_configuration() -> Dict[str, Any]:
    """
    Check device availability and return complete configuration.
    
    Returns:
        Device configuration dictionary
    """
    print("STEP 1: Checking Your System")
    print("=" * 60)
    print("\n🔍 What are we checking?")
    print("   We need to know if you have a GPU (graphics card) available.")
    print("   GPUs are much faster for AI, but CPUs work too (just slower).\n")
    print(f"   Python version: {sys.version.split()[0]}")
    
    is_gpu, gpu_name, cuda_version = check_gpu_availability()
    
    if is_gpu:
        print(f"\n   ✅ Great! GPU Available: {gpu_name}")
        print(f"   ✅ CUDA Version: {cuda_version}")
        print("   🚀 Your system will run much faster with GPU!")
        return create_gpu_config(gpu_name, cuda_version)
    else:
        print("\n   ⚠️  GPU NOT detected - using CPU")
        print("   💡 Don't worry! CPU works fine, it's just slower.")
        print("   💡 For faster results, consider using Google Colab (free GPU)")
        return create_cpu_config()


# ============================================================================
# AUTHENTICATION
# ============================================================================

def get_hf_token() -> str:
    """Get Hugging Face token from environment."""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print_token_setup_instructions()
        sys.exit(1)
    return token


def print_token_setup_instructions() -> None:
    """Print instructions for setting up Hugging Face token."""
    print("\n❌ Hugging Face token not found!")
    print("\nTo set up your token:")
    print("1. Get your token from: https://huggingface.co/settings/tokens")
    print("2. Create a .env file in this directory with:")
    print("   HUGGINGFACE_HUB_TOKEN=your_token_here")
    print("\n   OR")
    print("\n3. Set environment variable:")
    print("   export HUGGINGFACE_HUB_TOKEN=your_token_here")


def setup_token() -> str:
    """Setup and validate Hugging Face token."""
    token = get_hf_token()
    print("✅ Hugging Face token loaded successfully!")
    preview = f"{token[:10]}...{token[-4:]}" if len(token) > 14 else "****"
    print(f"   Token preview: {preview}")
    return token


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_tokenizer(model_id: str, hf_token: str) -> AutoTokenizer:
    """Load tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_id, token=hf_token)


def load_model_cpu(model_id: str, hf_token: str, torch_dtype: Any) -> AutoModelForCausalLM:
    """Load model for CPU execution."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    return model.to("cpu")


def load_model_gpu(model_id: str, hf_token: str, torch_dtype: Any, device_map: str) -> AutoModelForCausalLM:
    """Load model for GPU execution."""
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch_dtype,
        device_map=device_map,
    )


def load_model(model_id: str, hf_token: str, device_config: Dict[str, Any]) -> AutoModelForCausalLM:
    """
    Load model based on device configuration.
    
    Returns:
        Loaded model
    """
    if device_config["is_cpu"]:
        return load_model_cpu(model_id, hf_token, device_config["torch_dtype"])
    else:
        return load_model_gpu(
            model_id, hf_token, device_config["torch_dtype"], device_config["device_map"]
        )


def print_loading_header(model_id: str, device_config: Dict[str, Any]) -> None:
    """Print header information for model loading."""
    print("\n" + "=" * 60)
    print("STEP 2: Loading the AI Model")
    print("=" * 60)
    print("\n📚 What's happening?")
    print("   We're downloading and loading the AI model into memory.")
    print("   Think of it like opening a large book - it takes time to load all the pages!")
    print(f"\n📋 Model: {model_id}")
    print(f"📋 Device: {device_config['device']} (where the model will run)")
    print(f"📋 Data Type: {device_config['torch_dtype']} (how numbers are stored)")
    print(f"\n⏳ Loading tokenizer and model...")
    print(f"   ⏱️  Estimated time: {device_config['load_time_estimate']} on first run")
    print("   💡 Tip: This is slower the first time (downloading), faster after that!\n")


def print_loading_success() -> None:
    """Print success message after loading."""
    print("✅ Model and tokenizer loaded successfully!")
    print("   🎉 The AI model is now ready to generate text!\n")


# ============================================================================
# PIPELINE CREATION
# ============================================================================

def build_pipeline_kwargs(model: Any, tokenizer: Any, device_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build keyword arguments for pipeline creation."""
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": device_config["max_new_tokens"],
        "do_sample": device_config["do_sample"],
        "num_return_sequences": 1,
    }
    
    if device_config["top_k"] is not None:
        kwargs["top_k"] = device_config["top_k"]
    
    if device_config["device_map"] is not None:
        kwargs["device_map"] = device_config["device_map"]
    else:
        kwargs["device"] = device_config["pipeline_device"]
    
    return kwargs


def create_pipeline_from_model(model: Any, tokenizer: Any, device_config: Dict[str, Any]) -> Any:
    """Create text generation pipeline from loaded model."""
    print("=" * 60)
    print("Setting Up the Pipeline (Easy Way)")
    print("=" * 60)
    print("\n📚 What's happening?")
    print("   We're wrapping the model in a 'pipeline' - a simple interface")
    print("   that handles all the complex steps (tokenization, formatting, etc.)")
    print("   for us. It's like putting training wheels on a bike - easier to use!\n")
    
    pipeline_kwargs = build_pipeline_kwargs(model, tokenizer, device_config)
    chatbot = pipeline("text-generation", **pipeline_kwargs)
    
    print("✅ Pipeline created and ready to use!\n")
    return chatbot


# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_with_pipeline(chatbot: Any, messages: List[Dict[str, str]]) -> None:
    """Generate response using pipeline approach."""
    print("=" * 60)
    print("Generating AI Response (Pipeline Method)")
    print("=" * 60)
    print("\n📚 What's happening?")
    print("   The pipeline is processing your message and generating a response.")
    print("   It's doing all the work automatically - tokenization, model inference,")
    print("   and decoding - all in one simple function call!\n")
    
    start_time = time.time()
    with torch.no_grad():
        result = chatbot(messages)
    generation_time = time.time() - start_time
    
    print(f"✅ Response generated in {generation_time:.2f} seconds")
    print(f"   ⚡ That's pretty fast for AI text generation!\n")
    
    print_result(result)
    print_clean_response(result)


def prepare_model_inputs(tokenizer: Any, conversation: List[Dict[str, str]], model: Any) -> Dict[str, Any]:
    """Prepare inputs for direct model generation."""
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(model.device)


def generate_with_direct_model(model: Any, tokenizer: Any, device_config: Dict[str, Any], conversation: List[Dict[str, str]]) -> None:
    """Generate response using direct model approach."""
    print("=" * 60)
    print("Direct Model Approach: Generating Response")
    print("=" * 60)
    
    inputs = prepare_model_inputs(tokenizer, conversation, model)
    print("Generating response...\n")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=device_config["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id,
            do_sample=device_config["do_sample"],
        )
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"✅ Response generated in {generation_time:.2f} seconds\n")
    print("=" * 60)
    print("Response:")
    print(response)
    print("=" * 60)
    print()


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_result(result: Any) -> None:
    """Print full result in JSON format."""
    print("=" * 60)
    print("Full Result (Raw Output):")
    print("   💡 This shows the complete response structure with all metadata")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


def print_clean_response(result: Any) -> None:
    """Print clean assistant response."""
    assistant_reply = result[0]["generated_text"][-1]["content"]
    print("\n" + "=" * 60)
    print("Clean Assistant Response (Just the Text):")
    print("   💡 This is the actual response text, cleaned up and easy to read")
    print("=" * 60)
    print(assistant_reply)
    print("=" * 60)
    print()


# ============================================================================
# MODEL SELECTION & DISPLAY
# ============================================================================

def get_model_info(model_key: str) -> Tuple[Dict[str, Any], str]:
    """Get model information and ID."""
    model_info = AVAILABLE_MODELS[model_key]
    model_id = model_info["model_id"]
    return model_info, model_id


def print_model_info(model_info: Dict[str, Any], device_config: Dict[str, Any]) -> None:
    """Print model and device information."""
    print(f"\n📋 Selected Model: {model_info['name']}")
    print(f"📋 Description: {model_info['description']}")
    print(f"📋 Device: {device_config['device']}")
    
    if device_config["is_cpu"]:
        print(f"📋 Estimated RAM needed: ~{model_info['cpu_ram_required']}GB")
    else:
        print(f"📋 GPU: {device_config['gpu_name']}")
        print(f"📋 Estimated VRAM needed: ~{model_info['gpu_vram_required']}GB")
    
    print(f"📋 Recommended for: {model_info['recommended_for']}\n")


# ============================================================================
# ERROR HANDLING
# ============================================================================

def handle_memory_error(model_info: Dict[str, Any], device_config: Dict[str, Any]) -> None:
    """Handle memory error with helpful suggestions."""
    print("\n\n❌ Out of Memory Error!")
    
    if device_config["is_cpu"]:
        print(f"   {model_info['name']} needs ~{model_info['cpu_ram_required']}GB RAM")
    else:
        print(f"   {model_info['name']} needs ~{model_info['gpu_vram_required']}GB VRAM")
    
    print("   Solutions:")
    print("   1. Close other applications to free memory")
    print("   2. Use Google Colab with GPU (free)")
    print("   3. Try the Qwen model instead (change DEFAULT_MODEL to 'qwen')")
    
    if device_config["is_cpu"]:
        print("   4. Check available memory: free -h")


def handle_killed_process(model_info: Dict[str, Any], device_config: Dict[str, Any]) -> None:
    """Handle killed process (out of memory) error."""
    print("\n\n❌ Process was killed (out of memory)")
    
    if device_config["is_cpu"]:
        print(f"   {model_info['name']} needs ~{model_info['cpu_ram_required']}GB RAM")
    else:
        print(f"   {model_info['name']} needs ~{model_info['gpu_vram_required']}GB VRAM")
    
    print("   Solutions:")
    print("   1. Close other applications")
    print("   2. Use Google Colab with GPU")
    print("   3. Try the Qwen model instead (change DEFAULT_MODEL to 'qwen')")


def handle_general_error(error: Exception) -> None:
    """Handle general errors."""
    print(f"\n❌ Error: {error}")
    import traceback
    traceback.print_exc()


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_pipeline_example(
    model: Any, 
    tokenizer: Any, 
    device_config: Dict[str, Any],
    system_prompt: str,
    user_prompt: str
) -> None:
    """
    Run pipeline-based generation example.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        device_config: Device configuration
        system_prompt: System role/persona prompt
        user_prompt: User question/prompt
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    chatbot = create_pipeline_from_model(model, tokenizer, device_config)
    generate_with_pipeline(chatbot, messages)


def run_direct_model_example(
    model: Any, 
    tokenizer: Any, 
    device_config: Dict[str, Any],
    user_prompt: str
) -> None:
    """
    Run direct model generation example.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        device_config: Device configuration
        user_prompt: User question/prompt
    """
    conversation = [
        {"role": "user", "content": user_prompt}
    ]
    generate_with_direct_model(model, tokenizer, device_config, conversation)




def main() -> None:
    """Main orchestration function."""
    print("=" * 60)
    print("Prompt Engineering Basics")
    print("=" * 60)
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    device_config = get_device_configuration()
    hf_token = setup_token()
    model_info, model_id = get_model_info(DEFAULT_MODEL)
    print_model_info(model_info, device_config)
    
    try:
        print_loading_header(model_id, device_config)
        tokenizer = load_tokenizer(model_id, hf_token)
        model = load_model(model_id, hf_token, device_config)
        print_loading_success()
        
        # ====================================================================
        # UNDERSTANDING THE TWO APPROACHES
        # ====================================================================
        print("\n" + "=" * 60)
        print("📖 Learning: Two Ways to Use AI Models")
        print("=" * 60)
        print("\n🎓 As a student, it's important to understand both approaches:")
        print("\n" + "─" * 60)
        print("📦 METHOD 1: Pipeline Approach (The Easy Way)")
        print("─" * 60)
        print("   Think of it like: Using a vending machine")
        print("   • You put in your request (message)")
        print("   • The machine does everything automatically")
        print("   • You get your result (response)")
        print("\n   ✅ Pros: Simple, fast to code, less error-prone")
        print("   ❌ Cons: Less control, can't customize much")
        print("   🎯 Best for: Learning, quick tests, simple projects")
        print("\n" + "─" * 60)
        print("🔧 METHOD 2: Direct Model Approach (The Detailed Way)")
        print("─" * 60)
        print("   Think of it like: Cooking from scratch")
        print("   • You prepare ingredients (tokenize text)")
        print("   • You cook step by step (run model)")
        print("   • You plate the food (decode response)")
        print("\n   ✅ Pros: Full control, can customize everything")
        print("   ❌ Cons: More code, more things that can go wrong")
        print("   🎯 Best for: Advanced projects, research, custom needs")
        print("\n" + "=" * 60)
        print("💡 Key Takeaway:")
        print("   Pipeline = Easy but less control (like using a library function)")
        print("   Direct   = More work but full control (like writing your own function)")
        print("=" * 60 + "\n")
        
        # ====================================================================
        # EXAMPLE 1: Pipeline Approach
        # ====================================================================
        # PROMPT CONFIGURATION - Right before use for clarity
        SYSTEM_PROMPT = "You are Tom and I am Jerry"  # System prompt: Sets the role/persona
        USER_PROMPT_1 = "Who are you?"  # User prompt: Question from the user
        
        print("\n" + "=" * 60)
        print("EXAMPLE 1: Pipeline Approach (The Easy Way)")
        print("=" * 60)
        print("\n📝 What we're asking the AI:")
        print(f"   System Role: {SYSTEM_PROMPT}")
        print(f"   User Question: {USER_PROMPT_1}")
        print("\n💡 Remember: Pipeline does all the work automatically!")
        print("=" * 60 + "\n")
        
        run_pipeline_example(
            model, 
            tokenizer, 
            device_config,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT_1
        )
        
        # ====================================================================
        # EXAMPLE 2: Direct Model Approach
        # ====================================================================
        # PROMPT CONFIGURATION - Right before use for clarity
        USER_PROMPT_2 = "What's the weather like in Paris?"  # User prompt: Question from the user
        
        print("\n" + "=" * 60)
        print("EXAMPLE 2: Direct Model Approach (The Detailed Way)")
        print("=" * 60)
        print("\n📝 What we're asking the AI:")
        print(f"   User Question: {USER_PROMPT_2}")
        print("   (No system prompt in this example)")
        print("\n💡 Remember: We're doing each step manually for full control!")
        print("=" * 60 + "\n")
        
        run_direct_model_example(
            model, 
            tokenizer, 
            device_config,
            user_prompt=USER_PROMPT_2
        )
        
        # ====================================================================
        # COMPLETION
        # ====================================================================
        print("=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\n💡 Note: Model was loaded once and reused for both approaches.")
        print("   This is more efficient than loading the model multiple times.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except MemoryError:
        handle_memory_error(model_info, device_config)
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        if "Killed" in error_msg or "killed" in error_msg:
            handle_killed_process(model_info, device_config)
        else:
            handle_general_error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
