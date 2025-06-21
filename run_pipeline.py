#!/usr/bin/env python3
"""
Main entry point wrapper for the sentiment analysis pipeline.

This script provides an easy way to run the complete sentiment analysis pipeline
without needing to navigate to the src directory.

Usage:
    python run_pipeline.py           # Run complete pipeline
    python run_pipeline.py --help    # Show options
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the pipeline
try:
    from clean_pipeline import CleanNLPPipeline
    from config import setup_environment, RANDOM_SEED
    from utils import print_system_info, create_directory_structure
    
    def main():
        """Run the complete sentiment analysis pipeline."""
        print("🚀 Sentiment Analysis Pipeline")
        print("=" * 50)
        print_system_info()
        
        # Setup environment
        setup_environment()
        create_directory_structure()
        
        # Run pipeline
        pipeline = CleanNLPPipeline(seed=RANDOM_SEED)
        results = pipeline.run_complete_pipeline()
        
        if results:
            print(f"\n🌟 SUCCESS: Pipeline completed with {results['evaluation']['accuracy']*100:.1f}% accuracy!")
            print("📋 Check reports/ directory for detailed analysis")
            return 0
        else:
            print("\n❌ Pipeline failed")
            return 1

    if __name__ == "__main__":
        exit_code = main()
        sys.exit(exit_code)

except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("🔄 Pipeline will run in simulation mode...")
    try:
        # Try to run anyway
        from clean_pipeline import CleanNLPPipeline
        from config import setup_environment, RANDOM_SEED
        from utils import print_system_info, create_directory_structure
        
        def main():
            """Run the complete sentiment analysis pipeline."""
            print("🚀 Sentiment Analysis Pipeline (Simulation Mode)")
            print("=" * 50)
            print_system_info()
            
            setup_environment()
            create_directory_structure()
            
            pipeline = CleanNLPPipeline(seed=RANDOM_SEED)
            results = pipeline.run_complete_pipeline()
            
            if results:
                print(f"\n🌟 SUCCESS: Pipeline completed with {results['evaluation']['accuracy']*100:.1f}% accuracy!")
                print("📋 Check reports/ directory for detailed analysis")
                return 0
            else:
                print("\n❌ Pipeline failed")
                return 1

        if __name__ == "__main__":
            exit_code = main()
            sys.exit(exit_code)
            
    except Exception as inner_e:
        print(f"❌ Critical error: {inner_e}")
        print("Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1) 