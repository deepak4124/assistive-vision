"""
Verify if the Qwen2.5-VL-3B-Instruct model is accessible with the given HF token.
"""

import os
from huggingface_hub import snapshot_download, model_info

# Read the HF token from the environment
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Export it before running this script.")

print(f"Testing access to model: {MODEL_NAME}")
print(f"Using HF Token: {HF_TOKEN[:6]}...{HF_TOKEN[-4:]}")
print("-" * 60)

try:
    # Step 1: Check if model exists and get info
    print("\n1. Fetching model info...")
    info = model_info(MODEL_NAME, token=HF_TOKEN)
    print(f"   ✓ Model found: {info.id}")
    print(f"   ✓ Private: {info.private}")
    print(f"   ✓ Downloads: {info.downloads}")
    
    # Step 2: Try to download model files
    print("\n2. Attempting to download model files...")
    print("   (This may take a few minutes on first run)")
    cache_dir = snapshot_download(
        MODEL_NAME,
        token=HF_TOKEN,
        local_files_only=False
    )
    print(f"   ✓ Model downloaded to: {cache_dir}")
    
    # Step 3: Verify files
    print("\n3. Verifying model files...")
    import os
    files = os.listdir(cache_dir)
    print(f"   ✓ Found {len(files)} files:")
    for file in sorted(files)[:10]:  # Show first 10 files
        print(f"      - {file}")
    if len(files) > 10:
        print(f"      ... and {len(files) - 10} more files")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS: Model is accessible and ready for deployment!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    print("\nPossible issues:")
    print("  - Invalid HF token")
    print("  - Model doesn't exist or is private")
    print("  - Network connectivity issue")
    print("  - Insufficient permissions")
    print("\n" + "=" * 60)
    print("✗ FAILED: Model verification unsuccessful")
    print("=" * 60)
    exit(1)
