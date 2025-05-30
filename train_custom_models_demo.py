#!/usr/bin/env python3
"""
🚀 Custom ML Model Training Demo for CRM System

This script demonstrates training custom machine learning models
using your database data to enhance candidate matching beyond
the current AI system.
"""

import sys
import os
import logging
from datetime import datetime

# Add the app directory to Python path
sys.path.append('app')

from app.ml_model_trainer import CandidateMatchingModelTrainer, train_custom_models, generate_model_performance_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main demo function."""
    
    print("🧠 CRM AI ENHANCEMENT DEMO")
    print("="*60)
    print("Training custom machine learning models using your database...")
    print()
    
    try:
        # Initialize trainer
        print("📊 STEP 1: Loading Database Tables")
        print("-" * 40)
        
        trainer = CandidateMatchingModelTrainer(data_path="database_tables/")
        
        print(f"✅ Loaded {len(trainer.resources)} candidates")
        print(f"✅ Loaded {len(trainer.skills)} skills") 
        print(f"✅ Loaded {len(trainer.resource_skills)} skill mappings")
        print(f"✅ Loaded {len(trainer.experiences)} experience records")
        print(f"✅ Loaded {len(trainer.studies)} education records")
        print()
        
        # Create feature matrix
        print("🔧 STEP 2: Creating Feature Matrix")
        print("-" * 40)
        
        feature_matrix = trainer.create_candidate_feature_matrix()
        print(f"✅ Created feature matrix: {feature_matrix.shape[0]} candidates × {feature_matrix.shape[1]} features")
        
        # Show sample features
        print("\n📋 Sample Candidate Features:")
        sample_candidate = feature_matrix.iloc[0]
        for feature, value in sample_candidate.items():
            if feature != 'candidate_id':
                print(f"  {feature}: {value}")
        print()
        
        # Train models
        print("🤖 STEP 3: Training Custom ML Models")
        print("-" * 40)
        
        # Train skill similarity model
        print("Training Skill Similarity Model...")
        skill_results = trainer.train_skill_similarity_model()
        print(f"✅ Skill Model - R² Score: {skill_results['r2_score']:.4f}")
        print()
        
        # Train candidate matching model
        print("Training Candidate-Job Matching Model...")
        matching_results = trainer.train_candidate_job_matching_model()
        print(f"✅ Matching Model - R² Score: {matching_results['ensemble_r2']:.4f}")
        print(f"✅ Training Samples: {matching_results['training_samples']:,}")
        print(f"✅ Features Used: {matching_results['feature_count']}")
        print()
        
        # Save models
        print("💾 STEP 4: Saving Trained Models")
        print("-" * 40)
        
        trainer.save_models()
        print("✅ All models saved to 'trained_models/' directory")
        print()
        
        # Test predictions
        print("🧪 STEP 5: Testing Custom Predictions")
        print("-" * 40)
        
        # Test with sample candidate and job
        test_candidate_features = {
            'years_of_experience': 5,
            'total_skills': 12,
            'programming_skills': 3,
            'framework_skills': 4,
            'database_skills': 2,
            'cloud_skills': 1,
            'highest_degree_level': 2,
            'cs_related_degrees': 1
        }
        
        test_job_requirements = {
            'required_years_experience': 4,
            'required_skills': 10,
            'required_programming_skills': 2,
            'required_framework_skills': 3,
            'required_database_skills': 1,
            'required_degree_level': 2,
            'seniority_level': 'mid'
        }
        
        match_score = trainer.predict_match_score(
            test_candidate_features, 
            test_job_requirements
        )
        
        print(f"🎯 Sample Prediction:")
        print(f"  Candidate Profile: {test_candidate_features}")
        print(f"  Job Requirements: {test_job_requirements}")
        print(f"  🔥 Match Score: {match_score:.1%}")
        print()
        
        # Performance report
        print("📈 STEP 6: Model Performance Report")
        print("-" * 40)
        
        report = trainer.get_model_performance_report()
        
        print(f"📊 Data Statistics:")
        for key, value in report['data_statistics'].items():
            print(f"  {key}: {value:,}")
        
        print(f"\n🤖 Trained Models ({len(report['trained_models'])}):")
        for model in report['trained_models']:
            print(f"  ✓ {model}")
        
        print(f"\n💡 Enhanced Capabilities:")
        for capability in report['model_capabilities']:
            print(f"  • {capability}")
        
        print()
        
        # Integration guide
        print("🔗 STEP 7: Integration Guide")
        print("-" * 40)
        print("Your custom models are now ready! Here's how to integrate them:")
        print()
        print("1. 📁 Models saved in: 'trained_models/' directory")
        print("2. 🔧 Load models using: joblib.load('trained_models/model_name.joblib')")
        print("3. 🚀 Use in CandidateMatcher class for enhanced scoring")
        print("4. ⚡ Combine with existing AI search for best results")
        print()
        
        # Comparison with current system
        print("🆚 STEP 8: Enhancement Over Current System")
        print("-" * 40)
        print("Your CURRENT AI system already has:")
        print("  ✅ Semantic skill matching (Sentence Transformers)")
        print("  ✅ Multi-dimensional scoring (Skills, Experience, Education)")
        print("  ✅ Intelligent job parsing (OpenAI GPT)")
        print("  ✅ Enhanced search pipeline")
        print()
        print("NEW Custom Models ADD:")
        print("  🔥 Data-specific patterns from YOUR 79 candidates")
        print("  🔥 Skill co-occurrence learning from YOUR domain")
        print("  🔥 Custom feature engineering for YOUR use cases")
        print("  🔥 Ensemble predictions for higher accuracy")
        print("  🔥 Continuous learning capability")
        print()
        
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        print("Your CRM now has custom ML models trained on your specific data!")
        print("This enhances the existing AI system with domain-specific intelligence.")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        return False


def test_model_loading():
    """Test loading and using saved models."""
    
    print("\n🧪 BONUS: Testing Model Loading")
    print("-" * 40)
    
    import joblib
    
    try:
        # Load a saved model
        model_path = "trained_models/matching_random_forest.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Successfully loaded: {model_path}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Feature count: {model.n_features_in_}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
        
        # Load metadata
        metadata_path = "trained_models/metadata.json"
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Model metadata loaded:")
            print(f"   Created: {metadata['created_at']}")
            print(f"   Models: {len(metadata['models'])}")
            print(f"   Based on: {metadata['candidate_count']} candidates")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")


def show_integration_example():
    """Show how to integrate custom models with existing system."""
    
    print("\n🔧 INTEGRATION EXAMPLE")
    print("-" * 40)
    
    integration_code = '''
# Integration with existing CandidateMatcher
from app.candidate_matcher import CandidateMatcher
import joblib

class EnhancedCandidateMatcher(CandidateMatcher):
    def __init__(self):
        super().__init__()
        # Load custom models
        self.custom_rf_model = joblib.load('trained_models/matching_random_forest.joblib')
        self.custom_gb_model = joblib.load('trained_models/matching_gradient_boosting.joblib')
        self.custom_scaler = joblib.load('trained_models/scaler_matching.joblib')
    
    def enhanced_scoring(self, candidate, job_requirements):
        # Get current AI score
        current_score = self.find_matching_candidates_enhanced(job_requirements)
        
        # Get custom ML score
        candidate_features = self.extract_candidate_features(candidate)
        job_features = self.extract_job_features(job_requirements)
        
        # Combine features and predict
        combined_features = self.combine_features(candidate_features, job_features)
        ml_score = self.predict_with_custom_models(combined_features)
        
        # Ensemble: 70% current AI + 30% custom ML
        final_score = 0.7 * current_score + 0.3 * ml_score
        
        return final_score
'''
    
    print("Example integration code:")
    print(integration_code)


if __name__ == "__main__":
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = main()
    
    if success:
        test_model_loading()
        show_integration_example()
        
        print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🎯 Next Steps:")
        print("1. Review the trained models in 'trained_models/' directory")
        print("2. Integrate custom models with your existing CandidateMatcher")
        print("3. Test with real job descriptions in your chat interface")
        print("4. Monitor performance and retrain as you get more data")
    else:
        print("\n❌ Training failed. Check the error messages above.") 