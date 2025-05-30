#!/usr/bin/env python3
"""
Quick Training Script for ML Candidate Matching

This script trains the ML model using your existing candidate data.
Run this before using the AI search feature.
"""

import os
import sys
import logging

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from ai_candidate_matching.services.matching_service import MLMatchingService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Train the ML model with your candidate data."""
    try:
        logger.info("🚀 Starting AI Candidate Matching Model Training...")
        
        # Initialize the ML service
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database_tables")
        model_path = os.path.join(os.path.dirname(__file__), "models")
        
        logger.info(f"📂 Data path: {data_path}")
        logger.info(f"🤖 Model path: {model_path}")
        
        # Check data files
        required_files = ['resources.csv', 'skills.csv', 'resource_skills.csv', 'experiences.csv']
        for file in required_files:
            file_path = os.path.join(data_path, file)
            if not os.path.exists(file_path):
                logger.error(f"❌ Required file not found: {file_path}")
                return False
        
        logger.info("✅ All required data files found")
        
        # Initialize ML service
        ml_service = MLMatchingService(data_path=data_path, model_path=model_path)
        
        # Train the model (quick training for faster results)
        logger.info("🧠 Training ML model...")
        results = ml_service.train_model(
            num_synthetic_jobs=100,    # Reduced for speed
            num_training_pairs=1000,   # Reduced for speed
            validation_split=0.2,
            optimize_hyperparameters=False  # Skip for speed
        )
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Candidates: {results['num_candidates']}")
        logger.info(f"   - Training pairs: {results['num_training_pairs']}")
        logger.info(f"   - Features: {results['feature_dimensions']}")
        
        if 'training_metrics' in results:
            metrics = results['training_metrics']
            logger.info(f"   - Model accuracy: {metrics.get('accuracy', 0):.2%}")
            logger.info(f"   - R² score: {metrics.get('r2_score', 0):.3f}")
        
        logger.info("✅ Model is now ready for use in the web app!")
        logger.info("🌐 You can now use the AI Search feature in your Flask app")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 SUCCESS! Your AI matching system is ready!")
        print("💡 Now restart your Flask app and try the AI Search feature")
    else:
        print("\n❌ Training failed. Check the logs above for details.") 