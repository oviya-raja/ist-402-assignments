#!/usr/bin/env python3
"""
Prompt Engineering Basics - Standalone Python Script
Extracted from W3__Prompt_Engineering_Basics.ipynb

This script demonstrates:
- Basic prompt engineering with Qwen3-0.6B (Latest Qwen3 series)
- Model interaction using Hugging Face Transformers
- Pipeline vs Direct Model Loading
- Device optimization (CPU/GPU)
"""

import sys
import os
import time
import json
from pathlib import Path

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Or set HUGGINGFACE_HUB_TOKEN environment variable directly\n")

# Import required libraries
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print("❌ Required packages not installed!")
    print("   Install with: pip install transformers torch sentence-transformers")
    sys.exit(1)


def check_environment():
    """Check Python version and device availability"""
    print("🔍 Checking environment...")
    print(f"   Python version: {sys.version.split()[0]}")

    # Check GPU availability
    try:
        if torch.cuda.is_available():
            print(f"   ✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA Version: {torch.version.cuda}")
            return "cuda"
        else:
            print("   ⚠️  GPU NOT detected - using CPU")
            return "cpu"
    except Exception as e:
        print(f"   ⚠️  Error checking GPU: {e}")
        return "cpu"


def setup_token():
    """Setup Hugging Face token from environment or .env file"""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    if not token:
        print("\n❌ Hugging Face token not found!")
        print("\nTo set up your token:")
        print("1. Get your token from: https://huggingface.co/settings/tokens")
        print("2. Create a .env file in this directory with:")
        print("   HUGGINGFACE_HUB_TOKEN=your_token_here")
        print("\n   OR")
        print("\n3. Set environment variable:")
        print("   export HUGGINGFACE_HUB_TOKEN=your_token_here")
        print("\n   OR")
        print("\n4. Set it directly in this script (not recommended for sharing)")
        sys.exit(1)

    print("✅ Hugging Face token loaded successfully!")
    print(
        f"   Token preview: {token[:10]}...{token[-4:] if len(token) > 14 else '****'}"
    )
    return token


def example_1_pipeline_approach(model_id, hf_token, device):
    """Example 1: Using pipeline for text generation"""
    print("\n" + "=" * 60)
    print("Example 1: Using Pipeline Approach")
    print("=" * 60)

    # Configure device-optimized parameters
    if device == "cpu":
        print("   ✅ Using Qwen3-0.6B (~1.2GB disk, ~2.5GB RAM)")
        print("   ⏱️  Expected load time: 1-2 minutes")
        print("   ⏱️  Expected generation: 3-8 seconds per response")
        torch_dtype = torch.float32
        max_new_tokens = 256
        # For CPU, don't use device_map (requires accelerate and may cause issues)
        device_map = None
        pipeline_device = -1  # -1 means CPU in pipeline
    else:  # GPU
        print("   ⏱️  Expected load time: 30-60 seconds")
        print("   ⏱️  Expected generation: 1-2 seconds per response")
        torch_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        max_new_tokens = 512
        device_map = "auto"
        pipeline_device = 0  # 0 means first GPU

    print(f"   Device: {device}")
    print(f"   Torch dtype: {torch_dtype}")
    print(f"   📦 Model size: ~1.2GB disk, ~2.5GB RAM (will download on first run)\n")

    # Create a conversation with system prompt and user message
    messages = [
        {"role": "system", "content": "You are Tom and I am Jerry"},
        {"role": "user", "content": "Who are you?"},
    ]

    print("⏳ Loading Qwen3-0.6B model...")
    print("   This should be quick (~1-2 minutes on first run)...\n")

    # Set up the text generation pipeline
    pipeline_kwargs = {
        "model": model_id,
        "token": hf_token,
        "dtype": torch_dtype,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k": 10,
        "num_return_sequences": 1,
        "eos_token_id": 2,
    }

    # Add device-specific parameters
    if device_map:
        pipeline_kwargs["device_map"] = device_map
    else:
        pipeline_kwargs["device"] = pipeline_device

    chatbot = pipeline("text-generation", **pipeline_kwargs)

    print("✅ Model loaded! Generating response...\n")

    # Generate response
    start_time = time.time()
    result = chatbot(messages)
    generation_time = time.time() - start_time

    print(f"✅ Response generated in {generation_time:.2f} seconds\n")
    print("=" * 60)
    print("Full Result:")
    print(json.dumps(result, indent=2))
    print("=" * 60)

    # Extract just the assistant's response
    assistant_reply = result[0]["generated_text"][-1]["content"]
    print("\n" + "=" * 60)
    print("Clean Assistant Response:")
    print(assistant_reply)
    print("=" * 60)

    return chatbot


def example_2_direct_model_loading(model_id, hf_token, device):
    """Example 2: Using direct model loading"""
    print("\n" + "=" * 60)
    print("Example 2: Direct Model Loading Approach")
    print("=" * 60)

    print("⏳ Loading tokenizer and model...")
    print("   This should be quick (~1-2 minutes on first run)...\n")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    # Load the model
    if device == "cpu":
        torch_dtype = torch.float32
        # For CPU, don't use device_map (requires accelerate)
        # Use low_cpu_mem_usage to optimize memory
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,  # Optimize memory usage
        )
        model = model.to("cpu")
    else:
        torch_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=hf_token, dtype=torch_dtype, device_map="auto"
        )

    print("✅ Model loaded!\n")

    # Create a simple conversation
    conversation = [{"role": "user", "content": "What's the weather like in Paris?"}]

    # Convert conversation to model format
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    print("Generating response...\n")

    # Generate response
    outputs = model.generate(
        **inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id
    )

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("=" * 60)
    print("Response:")
    print(response)
    print("=" * 60)

    return model, tokenizer


def main():
    """Main function"""
    print("=" * 60)
    print("Prompt Engineering Basics")
    print("Qwen3-0.6B Example (Latest Qwen3 Series)")
    print("=" * 60)

    # Step 1: Check environment
    device = check_environment()

    # Step 2: Setup token
    hf_token = setup_token()

    # Step 3: Model ID
    # Using Qwen3-0.6B (latest Qwen3 series, works on low RAM systems)
    # Mistral-7B commented out - requires ~14GB RAM on CPU (too large for 7.65GB system)
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"  # Requires ~14GB RAM on CPU

    # Latest Qwen3-0.6B model (newest generation, works on your system)
    model_id = "Qwen/Qwen3-0.6B"

    print(f"\n📋 Model: {model_id}")
    print(f"📋 Device: {device}")
    if device == "cpu":
        print(f"📋 Estimated RAM needed: ~2.5GB (you have ~3.2GB available) ✅\n")
    else:
        print(
            f"📋 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n"
        )

    # Step 4: Run examples
    try:
        # Example 1: Pipeline approach
        chatbot = example_1_pipeline_approach(model_id, hf_token, device)

        # Example 2: Direct model loading
        model, tokenizer = example_2_direct_model_loading(model_id, hf_token, device)

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except MemoryError:
        print("\n\n❌ Out of Memory Error!")
        print("   Qwen3-0.6B needs ~2.5GB RAM")
        print("   Solutions:")
        print("   1. Close other applications to free RAM")
        print("   2. Use Google Colab with GPU (free)")
        print("   3. Check available memory: free -h")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        if "Killed" in error_msg or "killed" in error_msg:
            print("\n\n❌ Process was killed (out of memory)")
            print("   Qwen3-0.6B needs ~2.5GB RAM")
            print("   Solutions:")
            print("   1. Close other applications")
            print("   2. Use Google Colab with GPU")
            print(
                "   3. You have ~3.2GB available - should be enough, check other processes"
            )
        else:
            print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
