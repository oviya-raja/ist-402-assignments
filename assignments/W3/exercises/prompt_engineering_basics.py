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
    print("🔍 Checking environment...")
    print(f"   Python version: {sys.version.split()[0]}")
    
    is_gpu, gpu_name, cuda_version = check_gpu_availability()
    
    if is_gpu:
        print(f"   ✅ GPU Available: {gpu_name}")
        print(f"   ✅ CUDA Version: {cuda_version}")
        return create_gpu_config(gpu_name, cuda_version)
    else:
        print("   ⚠️  GPU NOT detected - using CPU")
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
    print("Loading Model and Tokenizer")
    print("=" * 60)
    print(f"📋 Model: {model_id}")
    print(f"📋 Device: {device_config['device']}")
    print(f"📋 Torch dtype: {device_config['torch_dtype']}")
    print(f"\n⏳ Loading tokenizer and model...")
    print(f"   This may take {device_config['load_time_estimate']} on first run...\n")


def print_loading_success() -> None:
    """Print success message after loading."""
    print("✅ Model and tokenizer loaded successfully!\n")


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
    print("Creating Pipeline from Loaded Model")
    print("=" * 60)
    
    pipeline_kwargs = build_pipeline_kwargs(model, tokenizer, device_config)
    chatbot = pipeline("text-generation", **pipeline_kwargs)
    
    print("✅ Pipeline created!\n")
    return chatbot


# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_with_pipeline(chatbot: Any, messages: List[Dict[str, str]]) -> None:
    """Generate response using pipeline approach."""
    print("=" * 60)
    print("Pipeline Approach: Generating Response")
    print("=" * 60)
    
    start_time = time.time()
    with torch.no_grad():
        result = chatbot(messages)
    generation_time = time.time() - start_time
    
    print(f"✅ Response generated in {generation_time:.2f} seconds\n")
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
    print("Full Result:")
    print(json.dumps(result, indent=2))
    print("=" * 60)


def print_clean_response(result: Any) -> None:
    """Print clean assistant response."""
    assistant_reply = result[0]["generated_text"][-1]["content"]
    print("\n" + "=" * 60)
    print("Clean Assistant Response:")
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
        print("=" * 60)
        print("Understanding Pipeline vs Direct Model Approaches")
        print("=" * 60)
        print("\n📦 PIPELINE APPROACH:")
        print("   • Uses Hugging Face's pipeline() wrapper")
        print("   • Higher-level abstraction - simpler to use")
        print("   • Automatically handles tokenization and formatting")
        print("   • Good for: Quick prototyping, simple use cases")
        print("   • Supports: System prompts + User prompts")
        print("   • Example: chatbot(messages) - just pass messages")
        print("\n🔧 DIRECT MODEL APPROACH:")
        print("   • Directly uses model.generate() method")
        print("   • Lower-level control - more flexibility")
        print("   • Manual tokenization and input preparation")
        print("   • Good for: Custom logic, fine-grained control")
        print("   • Supports: User prompts (can add system via chat template)")
        print("   • Example: model.generate(**inputs) - prepare inputs yourself")
        print("\n" + "=" * 60)
        print("Key Difference:")
        print("  Pipeline = Easy but less control")
        print("  Direct   = More work but full control")
        print("=" * 60 + "\n")
        
        # ====================================================================
        # EXAMPLE 1: Pipeline Approach
        # ====================================================================
        # PROMPT CONFIGURATION - Right before use for clarity
        SYSTEM_PROMPT = "You are Tom and I am Jerry"  # System prompt: Sets the role/persona
        USER_PROMPT_1 = "Who are you?"  # User prompt: Question from the user
        
        print("\n" + "=" * 60)
        print("EXAMPLE 1: Pipeline Approach (High-Level)")
        print("=" * 60)
        print("System Prompt:", SYSTEM_PROMPT)
        print("User Prompt:", USER_PROMPT_1)
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
        print("EXAMPLE 2: Direct Model Approach (Low-Level)")
        print("=" * 60)
        print("User Prompt:", USER_PROMPT_2)
        print("(No system prompt - using direct model.generate())")
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
