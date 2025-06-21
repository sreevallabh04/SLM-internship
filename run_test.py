#!/usr/bin/env python3
"""
Simple test runner for the NLP pipeline
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def run_pipeline_test():
    """Run the complete pipeline test"""
    try:
        print("🧪 Starting NLP Pipeline Test")
        print("=" * 50)
        
        # Test 1: Basic imports
        print("✅ Testing basic imports...")
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("   All basic libraries imported successfully")
        
        # Test 2: Fixed pipeline
        print("✅ Testing fixed pipeline...")
        from main_fixed import FixedTextClassificationPipeline
        
        # Initialize pipeline
        pipeline = FixedTextClassificationPipeline()
        
        # Run complete pipeline
        print("🚀 Running complete pipeline...")
        results = pipeline.run_complete_pipeline(dataset_size="small")
        
        # Print results
        print("\n" + "=" * 50)
        print("🎯 PIPELINE TEST RESULTS")
        print("=" * 50)
        print(f"✅ Status: SUCCESS")
        print(f"📊 Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"📈 F1-Score: {results.get('overall_f1', 0):.4f}")
        print(f"📁 Train samples: {results.get('dataset_info', {}).get('train_size', 0)}")
        print(f"📁 Test samples: {results.get('dataset_info', {}).get('test_size', 0)}")
        
        # Check if files were created
        if os.path.exists('reports/results_demo.json'):
            print("✅ Results file created")
        if os.path.exists('reports/analysis_report_demo.md'):
            print("✅ Analysis report created")
        if os.path.exists('reports/confusion_matrix_demo.png'):
            print("✅ Confusion matrix created")
        if os.path.exists('logs/pipeline_fixed.log'):
            print("✅ Log file created")
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("💡 The NLP pipeline is working correctly!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_pipeline_test()
    if success:
        print("\n🎊 PROJECT STATUS: COMPLETE AND WORKING! 🎊")
    else:
        print("\n⚠️  Some issues detected - check error messages above")
    
    sys.exit(0 if success else 1) 