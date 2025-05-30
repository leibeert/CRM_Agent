# AI-Powered Candidate Matching System

A sophisticated machine learning system for intelligent candidate-job matching based on skills, experience, and job requirements.

## 🎯 Overview

This system transforms traditional keyword-based candidate search into an intelligent ML-powered matching engine that:

- **Extracts 400+ features** from candidate profiles and job descriptions
- **Uses ensemble ML models** (Random Forest, Gradient Boosting, Neural Networks)
- **Provides explainable AI** with detailed compatibility explanations
- **Learns from feedback** to continuously improve matching accuracy
- **Handles semantic similarity** between skills and requirements

## 🏗️ Architecture

```
ai_candidate_matching/
├── core/
│   ├── feature_extractor.py    # Feature extraction from text and structured data
│   └── ml_models.py            # ML models for compatibility prediction
├── data/
│   ├── data_loader.py          # Data loading from CSV/database
│   └── data_preprocessor.py    # Data preprocessing utilities
├── services/
│   └── matching_service.py     # Main ML matching service
├── models/                     # Trained model storage
├── utils/                      # Utility functions
└── api/                        # API endpoints
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install scikit-learn pandas numpy joblib sqlalchemy pymysql
```

### 2. Train the Model

```bash
cd crm/app/ai_candidate_matching
python train_ml_model.py
```

### 3. Use the System

```python
from ai_candidate_matching.services.matching_service import MLMatchingService

# Initialize the service
ml_service = MLMatchingService(
    data_path="database_tables",
    model_path="app/ai_candidate_matching/models"
)

# Find best candidates for a job
candidates = ml_service.find_best_candidates(
    job_description="We need a Python developer with Django experience...",
    job_title="Senior Python Developer",
    required_experience=5,
    location="Paris, France",
    salary_range=(60000, 80000),
    top_k=10
)

# Get compatibility score for specific candidate-job pair
result = ml_service.predict_candidate_job_compatibility(
    candidate_data=candidate_profile,
    job_description="Job description here...",
    job_title="Software Engineer"
)
```

## 🧠 How It Works

### Feature Extraction

The system extracts **400+ features** from each candidate-job pair:

#### Text Features (300 features)
- TF-IDF vectors from job descriptions and candidate profiles
- Semantic analysis of requirements and qualifications

#### Skill Features (18 features)
- Skill category counts and ratios across 8 categories:
  - Programming Languages (Python, Java, JavaScript, etc.)
  - Web Frameworks (React, Django, Spring, etc.)
  - Databases (MySQL, PostgreSQL, MongoDB, etc.)
  - Cloud Platforms (AWS, Azure, GCP, etc.)
  - Data Science (ML, TensorFlow, pandas, etc.)
  - DevOps (Docker, Kubernetes, Jenkins, etc.)
  - Business Skills (Project Management, Agile, etc.)
  - Finance (Murex, Calypso, Trading, etc.)

#### Experience Features (11 features)
- Years of experience (raw and normalized)
- Experience level encoding (entry, junior, mid, senior, lead, expert)
- Experience bins and progression indicators

#### Title/Role Features (14 features)
- Role type indicators (developer, analyst, manager, etc.)
- Seniority indicators (senior, lead, principal, etc.)

#### Location Features (18 features)
- Tech hub indicators (Paris, London, Berlin, etc.)
- Remote work compatibility
- Country-specific preferences

#### Salary Features (4 features)
- Normalized salary ranges and spreads

#### Job Complexity Features (3 features)
- Technical complexity indicators
- Leadership requirements
- Innovation requirements

#### Career Progression Features (5 features)
- Upward progression indicators
- Job stability metrics
- Recent activity indicators

#### Skill Diversity Features (4 features)
- Skill breadth and depth metrics
- Category balance scores

### Machine Learning Models

The system uses an **ensemble approach** combining:

1. **Random Forest Regressor**
   - Handles non-linear relationships
   - Provides feature importance
   - Robust to outliers

2. **Gradient Boosting Regressor**
   - Sequential learning
   - High predictive accuracy
   - Handles complex patterns

3. **Neural Network (MLP)**
   - Deep feature interactions
   - Non-linear transformations
   - Adaptive learning

4. **Ridge Regression**
   - Linear baseline
   - Regularization
   - Interpretability

### Prediction Process

1. **Feature Extraction**: Convert candidate and job data into numerical features
2. **Ensemble Prediction**: Each model makes a prediction
3. **Score Aggregation**: Average predictions with confidence weighting
4. **Explanation Generation**: Analyze feature importance and generate human-readable explanations

## 📊 Performance Metrics

The system tracks multiple performance indicators:

- **Mean Squared Error (MSE)**: Prediction accuracy
- **R² Score**: Variance explained by the model
- **Mean Absolute Error (MAE)**: Average prediction error
- **Accuracy within thresholds**: Percentage of predictions within ±0.1, ±0.2, ±0.3

## 🎯 Key Features

### 1. Intelligent Skill Matching
- **Exact matches**: Direct skill alignment (100% weight)
- **Semantic similarity**: Related skills (80% weight)
- **Transferable skills**: Cross-domain skills (60% weight)
- **Skill categories**: Grouped skill analysis

### 2. Experience Analysis
- **Years matching**: Experience requirement alignment
- **Role progression**: Career advancement patterns
- **Industry relevance**: Domain-specific experience
- **Skill growth**: Learning trajectory analysis

### 3. Explainable AI
- **Score interpretation**: Human-readable compatibility levels
- **Key factors**: Top contributing factors to the score
- **Recommendations**: Actionable insights for improvement
- **Confidence scores**: Prediction reliability indicators

### 4. Continuous Learning
- **Feedback integration**: Learn from hiring outcomes
- **Model retraining**: Periodic model updates
- **Performance monitoring**: Track prediction accuracy over time

## 🔧 Configuration

### Data Sources
The system can load data from:
- **CSV files**: For development and testing
- **MySQL database**: For production deployment
- **SQLite**: For lightweight deployments

### Model Parameters
Customize model behavior:
```python
ml_service = MLMatchingService(
    data_path="path/to/data",
    model_path="path/to/models"
)

# Train with custom parameters
results = ml_service.train_model(
    num_synthetic_jobs=200,
    num_training_pairs=2000,
    validation_split=0.2,
    optimize_hyperparameters=True
)
```

## 📈 Training Data

The system creates synthetic training data by:

1. **Loading real candidate profiles** from your database
2. **Generating diverse job descriptions** using templates
3. **Creating candidate-job pairs** with compatibility scores
4. **Calculating synthetic scores** based on:
   - Skill matching (40% weight)
   - Experience alignment (30% weight)
   - Title/role compatibility (20% weight)
   - Random variation (10% weight)

## 🔍 Usage Examples

### Find Top Candidates
```python
top_candidates = ml_service.find_best_candidates(
    job_description="""
    We are seeking a Senior Python Developer to join our backend team.
    The ideal candidate will have 5+ years of experience with Django,
    PostgreSQL, and AWS. Experience with microservices and Docker is required.
    """,
    job_title="Senior Python Developer",
    required_experience=5,
    location="Paris, France",
    salary_range=(70000, 90000),
    top_k=5
)

for candidate in top_candidates:
    print(f"Score: {candidate['compatibility_score']:.3f}")
    print(f"Name: {candidate['candidate']['first_name']} {candidate['candidate']['last_name']}")
    print(f"Explanation: {candidate['explanation']['score_interpretation']}")
```

### Single Prediction
```python
result = ml_service.predict_candidate_job_compatibility(
    candidate_data={
        'first_name': 'John',
        'last_name': 'Doe',
        'title': 'Python Developer',
        'years_of_experience': 6,
        'skills': [
            {'name': 'Python', 'proficiency_level': 'advanced'},
            {'name': 'Django', 'proficiency_level': 'intermediate'},
            {'name': 'PostgreSQL', 'proficiency_level': 'intermediate'}
        ]
    },
    job_description="Senior Python Developer position...",
    job_title="Senior Python Developer",
    required_experience=5
)

print(f"Compatibility: {result['compatibility_score']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Model Performance
```python
metrics = ml_service.get_model_performance_metrics()
print(f"R² Score: {metrics['training_metrics']['ensemble_r2']:.3f}")
print(f"MAE: {metrics['training_metrics']['ensemble_mae']:.3f}")
```

## 🚀 Integration

### API Integration
```python
from ai_candidate_matching.api.ml_endpoints import MLMatchingAPI

# Create API endpoints
api = MLMatchingAPI(ml_service)

# Use in Flask/FastAPI
@app.route('/api/match-candidates', methods=['POST'])
def match_candidates():
    return api.find_candidates(request.json)
```

### Database Integration
```python
# Configure database connection
db_config = {
    'type': 'mysql',
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',
    'database': 'crm_db'
}

ml_service = MLMatchingService(db_config=db_config)
```

## 📊 Monitoring & Analytics

Track system performance:
- **Prediction accuracy**: Compare predictions with actual hiring outcomes
- **Feature importance**: Understand which factors matter most
- **Model drift**: Monitor performance degradation over time
- **User feedback**: Collect and integrate recruiter feedback

## 🔮 Future Enhancements

1. **Deep Learning Models**: Transformer-based models for better text understanding
2. **Real-time Learning**: Online learning from user interactions
3. **Multi-objective Optimization**: Balance multiple criteria (cost, time, quality)
4. **Bias Detection**: Ensure fair and unbiased candidate recommendations
5. **Market Intelligence**: Integrate salary and demand data
6. **Skill Embeddings**: Advanced semantic skill representations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 