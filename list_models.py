#!/usr/bin/env python3
"""
Debug script to list all models accessible with the current GEMINI_API_KEY.
Uses the new google-genai SDK.

Usage:
    python list_models.py
"""

import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    print("WARNING: python-dotenv not installed. Skipping .env file loading.")
    load_dotenv = None

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai SDK not installed. Install via: pip install google-genai")
    sys.exit(1)


def main():
    # Load .env file if python-dotenv is available
    if load_dotenv:
        load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment")
        sys.exit(1)

    try:
        # Initialize client
        print("Initializing Gemini client...")
        client = genai.Client(api_key=api_key)
        print("✓ Client initialized successfully\n")

        # List all available models
        print("Fetching available models...")
        models = client.models.list()
        print(f"✓ Retrieved {len(models)} model(s)\n")

        # Print each model name
        print("Available models:")
        print("-" * 60)
        for i, model in enumerate(models, 1):
            model_name = model.name if hasattr(model, "name") else str(model)
            print(f"{i}. {model_name}")
        print("-" * 60)

    except Exception as e:
        print(f"ERROR: Failed to list models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
