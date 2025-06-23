#!/usr/bin/env python3
"""
🚀 PyTorch-only Environment Setup

This script ensures a clean PyTorch-only environment for the 
DistilBERT sentiment analysis project, completely blocking TensorFlow.

Usage:
    python setup_pytorch_only.py
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def set_env_variables():
    """Set environment variables to block TensorFlow."""
    print("🚫 Setting TensorFlow-blocking environment variables...")
    
    env_vars = {
        'USE_TF': 'None',
        'USE_TORCH': '1',
        'TRANSFORMERS_VERBOSITY': 'error',
        'TOKENIZERS_PARALLELISM': 'false',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',  # For Mac compatibility
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    print("✅ Environment variables set")

def check_pytorch_installation():
    """Check if PyTorch is properly installed."""
    print("\n🔍 Checking PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        return True
    except ImportError:
        print("❌ PyTorch not found")
        return False

def check_transformers_installation():
    """Check if transformers[torch] is properly installed."""
    print("\n🔍 Checking Transformers installation...")
    
    try:
        # Import specific PyTorch components
        from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
        from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
        from transformers.trainer import Trainer
        from transformers.training_args import TrainingArguments
        
        print("✅ Transformers PyTorch components available")
        return True
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False

def check_tensorflow_blocked():
    """Verify TensorFlow is not imported."""
    print("\n🚫 Verifying TensorFlow is blocked...")
    
    # Check if TensorFlow is in sys.modules
    tf_modules = [mod for mod in sys.modules.keys() if 'tensorflow' in mod.lower()]
    
    if tf_modules:
        print(f"⚠️  TensorFlow modules found: {tf_modules}")
        return False
    
    # Try to import TensorFlow (should fail or be avoided)
    try:
        import tensorflow
        print("⚠️  TensorFlow was imported successfully (this may cause DLL issues)")
        return False
    except ImportError:
        print("✅ TensorFlow successfully blocked")
        return True
    except Exception as e:
        print(f"✅ TensorFlow blocked with error: {e}")
        return True

def install_pytorch_requirements():
    """Install PyTorch-only requirements."""
    print("\n📦 Installing PyTorch-only requirements...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--no-cache-dir"]
        
        print("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def verify_pytorch_only_imports():
    """Test importing core components without TensorFlow."""
    print("\n🧪 Testing PyTorch-only imports...")
    
    test_imports = [
        ("torch", "PyTorch core"),
        ("datasets", "Datasets library"),
        ("numpy", "NumPy"),
        ("sklearn.metrics", "Scikit-learn metrics"),
    ]
    
    all_success = True
    
    for module, description in test_imports:
        try:
            importlib.import_module(module)
            print(f"✅ {description}: OK")
        except ImportError as e:
            print(f"❌ {description}: FAILED - {e}")
            all_success = False
    
    # Test transformers specific imports
    try:
        from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
        from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
        print("✅ DistilBERT components: OK")
    except ImportError as e:
        print(f"❌ DistilBERT components: FAILED - {e}")
        all_success = False
    
    return all_success

def create_pytorch_test_script():
    """Create a test script to verify PyTorch-only functionality."""
    print("\n📝 Creating test script...")
    
    test_script = """#!/usr/bin/env python3
# PyTorch-only test script

import os
import sys

# Block TensorFlow
os.environ['USE_TF'] = 'None'
os.environ['USE_TORCH'] = '1'

try:
    import torch
    from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
    from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
    
    print("🎉 PyTorch-only test PASSED!")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Quick model test
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    print("✅ DistilBERT model loaded successfully")
    print("✅ All tests passed - ready for training!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
"""
    
    test_file = Path("test_pytorch_only.py")
    test_file.write_text(test_script)
    print(f"✅ Test script created: {test_file}")

def main():
    """Main setup function."""
    print("🚀 PyTorch-only Environment Setup")
    print("=" * 60)
    
    # Set environment variables
    set_env_variables()
    
    # Check current installations
    pytorch_ok = check_pytorch_installation()
    transformers_ok = check_transformers_installation()
    tensorflow_blocked = check_tensorflow_blocked()
    
    # Install requirements if needed
    if not pytorch_ok:
        print("\n📦 Installing PyTorch requirements...")
        install_pytorch_requirements()
    
    # Verify imports work
    imports_ok = verify_pytorch_only_imports()
    
    # Create test script
    create_pytorch_test_script()
    
    # Final status
    print("\n" + "=" * 60)
    print("🎯 SETUP SUMMARY")
    print("=" * 60)
    print(f"PyTorch installed: {'✅' if pytorch_ok else '❌'}")
    print(f"Transformers working: {'✅' if transformers_ok else '❌'}")
    print(f"TensorFlow blocked: {'✅' if tensorflow_blocked else '❌'}")
    print(f"All imports working: {'✅' if imports_ok else '❌'}")
    
    if all([pytorch_ok, transformers_ok, tensorflow_blocked, imports_ok]):
        print("\n🎉 SETUP SUCCESSFUL!")
        print("💡 You can now run: cd src && python train.py")
        print("💡 Or test with: python test_pytorch_only.py")
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("💡 Please fix the issues above before training")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 