# Colorectal Cancer Survival Prediction - MLOps Project

A complete MLOps pipeline for predicting colorectal cancer patient survival using machine learning. This project demonstrates end-to-end ML workflow including data processing, model training, web deployment, and Kubeflow pipeline orchestration.

## 🎯 Project Overview

This project predicts colorectal cancer patient survival based on various clinical and demographic factors including:
- Healthcare costs
- Tumor size
- Treatment type
- Diabetes status
- Mortality rate

The prediction model uses a Gradient Boosting Classifier to provide survival predictions through a user-friendly web interface.

## 🏗️ Architecture

```
├── Data Processing Pipeline
├── Model Training & Evaluation
├── Flask Web Application
├── Kubeflow Pipeline Orchestration
├── Docker Containerization
└── MLflow Experiment Tracking
```

## 📁 Project Structure

```
colorectal_mlops/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
├── Dockerfile                 # Container configuration
├── ml_pipeline.yaml           # Compiled Kubeflow pipeline
├── src/
│   ├── data_processing.py     # Data preprocessing pipeline
│   ├── model_trainer.py       # Model training pipeline
│   ├── logger.py             # Logging configuration
│   └── exception.py          # Custom exception handling
├── artifacts/
│   ├── models/               # Trained models
│   ├── processed/            # Processed datasets
│   └── raw/                  # Raw datasets
├── kubeflow_pipeline/
│   └── mlops_pipeline.py     # Kubeflow pipeline definition
├── templates/
│   └── index.html            # Web interface template
├── static/
│   └── style.css             # Web interface styling
└── notebooks/
    └── research.ipynb        # Exploratory data analysis
```

## 🚀 Features

- **Data Processing Pipeline**: Automated data cleaning, preprocessing, and feature engineering
- **Model Training**: Gradient Boosting Classifier with hyperparameter optimization
- **Web Interface**: Interactive Flask application for real-time predictions
- **MLOps Pipeline**: Kubeflow-based workflow orchestration
- **Experiment Tracking**: MLflow integration for model versioning and metrics
- **Containerization**: Docker support for easy deployment
- **Logging & Monitoring**: Comprehensive logging and error handling

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Docker (optional)
- Kubeflow (for pipeline execution)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd colorectal_mlops
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # OR
   pip install -e .
   ```

3. **Run data processing**
   ```bash
   python src/data_processing.py
   ```

4. **Train the model**
   ```bash
   python src/model_trainer.py
   ```

5. **Start the web application**
   ```bash
   python app.py
   ```

   The application will be available at `http://localhost:5000`

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t colorectal-mlops .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 colorectal-mlops
   ```

## 📊 Usage

### Web Interface

1. Navigate to `http://localhost:5000`
2. Fill in the patient information:
   - Healthcare Costs
   - Tumor Size (mm)
   - Mortality Rate per 100K
   - Treatment Type (numeric code)
   - Diabetes Status (0/1)
3. Click "Predict" to get the survival prediction

### Kubeflow Pipeline

1. **Compile the pipeline**
   ```bash
   python kubeflow_pipeline/mlops_pipeline.py
   ```

2. **Upload and run the pipeline** in your Kubeflow cluster using the generated `ml_pipeline.yaml`

## 🔧 Configuration

### Model Parameters
- **Algorithm**: Gradient Boosting Classifier
- **Estimators**: 100
- **Learning Rate**: 0.1
- **Max Depth**: 3
- **Random State**: 42

### Data Processing
- Automatic handling of categorical variables with Label Encoding
- Feature scaling using StandardScaler
- Train-test split with stratification

## 📈 Model Performance

The model is evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 🔍 Monitoring & Logging

- Comprehensive logging throughout the pipeline
- Custom exception handling for robust error management
- MLflow integration for experiment tracking
- Model versioning and artifact management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Taufeeq Ahmad**

## 🙏 Acknowledgments

- Healthcare data providers
- Open source ML community
- Kubeflow and MLflow teams

## 📞 Support

For support and questions, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and research purposes. Always consult healthcare professionals for medical decisions.