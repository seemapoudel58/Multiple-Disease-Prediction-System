# Multiple Disease Prediction System

## Overview

This project implements a comprehensive **Multiple Disease Prediction System** using Machine Learning algorithms. The system can predict three different diseases:

1. **Heart Disease** - Using Logistic Regression
2. **Diabetes** - Using Decision Tree
3. **Breast Cancer** - Using Support Vector Machine (SVM)

The application features a user-friendly Streamlit web interface with authentication, prediction history tracking, EDA visualizations, and detailed model descriptions.

## Features

- 🔐 **User Authentication**: Sign up and login system
- 📊 **Disease Predictions**: Predict Heart Disease, Diabetes, and Breast Cancer
- 📈 **Exploratory Data Analysis (EDA)**: Visualizations for each disease
- 📋 **Prediction History**: Track all your predictions in "My Predictions"
- 🧪 **Model Test Results**: View detailed performance metrics for each model
- 📚 **Model Descriptions**: Learn about the algorithms used (Logistic Regression, Decision Tree, SVM)
- 🎯 **User-Specific Tracking**: Personalized prediction history per user

## Datasets

### 1. Heart Disease Dataset

The dataset includes various medical features related to heart health from Kaggle.

#### Attribute Information

- **AGE**: Age in years
- **SEX**: (1 = male; 0 = female)
- **CP (Chest Pain Type)**:
  - 0: Typical angina (most serious)
  - 1: Atypical angina
  - 2: Non-anginal pain
  - 3: Asymptomatic (least serious)
- **TRESTBPS**: Resting blood pressure (in mm Hg on admission to the hospital)
- **CHOL**: Serum cholesterol in mg/dl
- **FBS**: (Fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- **RESTECG (Resting Electrocardiographic Results)**:
  - 0: Normal
  - 1: ST-T wave abnormality
  - 2: Left ventricular hypertrophy
- **THALACH**: Maximum heart rate achieved
- **EXANG**: Exercise-induced angina (1 = yes; 0 = no)
- **OLDPEAK**: ST depression induced by exercise relative to rest
- **SLOPE (The slope of the peak exercise ST segment)**:
  - 0: Upsloping
  - 1: Flat
  - 2: Downsloping
- **CA**: Number of major vessels (0-3) colored by fluoroscopy
- **THAL**:
  - 3 = Normal
  - 6 = Fixed defect
  - 7 = Reversible defect
- **TARGET**: Heart disease diagnosis (0 = No disease, 1 = Disease present)

### 2. Breast Cancer Dataset

The dataset uses the **Wisconsin Breast Cancer Dataset** from scikit-learn, containing features computed from digitized images of fine needle aspirates (FNA) of breast masses.

#### Key Features

The dataset includes 30 features describing characteristics of cell nuclei present in the image:

- **Mean Features**: Mean values of measurements (e.g., mean radius, mean texture, mean area, mean perimeter, etc.)
- **Standard Error Features**: Standard error of measurements
- **Worst Features**: Largest (worst) values of measurements

#### Top Features Used (Selected via PCA)

The model uses Principal Component Analysis (PCA) to identify the top 10 most important features for prediction, including:

- Mean radius, mean texture, mean area, mean perimeter
- Mean smoothness, mean compactness, mean concavity
- Mean concave points, mean symmetry, mean fractal dimension
- And other critical features

#### Target Variable

- **0**: Benign (Non-cancerous)
- **1**: Malignant (Cancerous)

## Models

- **Algorithm**: Decision Tree
- **Splitting Criteria**: Gini Impurity / Information Gain
- **Features**: 8 medical attributes
- **Performance**: Effective in capturing non-linear relationships

#### How It Works

Decision Tree recursively splits data based on feature values to form a tree structure. Each node represents a decision, and each leaf represents the predicted class (diabetes or not).

### 3. Breast Cancer Prediction - Support Vector Machine (SVM)

- **Algorithm**: Support Vector Machine (SVM)
- **Optimization**: Gradient Descent
- **Loss Function**: Hinge Loss
- **Dimensionality Reduction**: Principal Component Analysis (PCA) - 10 components
- **Features**: 30 original features reduced to top 10 via PCA
- **Performance**: High accuracy in classifying malignant vs benign tumors

#### How It Works

SVM finds the optimal hyperplane that best separates malignant and benign tumors by maximizing the margin between support vectors. The model:

- Uses PCA to reduce 30 features to 10 principal components
- Converts labels from {0, 1} to {-1, +1} for hinge loss calculation
- Trains using gradient descent with regularization
- Predicts based on which side of the hyperplane the data point falls

#### Key Advantages for Breast Cancer

- **High-Dimensional Data Handling**: Effective with 30 features
- **Robust Decision Boundary**: Maximizes margin for better generalization
- **Memory Efficient**: Only uses support vectors for prediction
- **Works Well with PCA**: Complements dimensionality reduction techniques

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/asmriti/Multiple-Disease-Prediction.git
   cd Multiple-Disease-Prediction
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv disease-prediction-env
   source disease-prediction-env/bin/activate  # On Windows: disease-prediction-env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser at `http://localhost:8501`

3. **First Time Users**:

   - Click "Sign Up" to create an account
   - Enter your email, username, and password
   - After signing up, you'll be automatically logged in

4. **Returning Users**:

   - Click "Login" and enter your credentials

5. **Making Predictions**:

   - Select a disease from the sidebar (Diabetes, Heart Disease, or Breast Cancer)
   - Enter the required patient details
   - Click the prediction button to get results
   - Your prediction will be saved to "My Predictions"

6. **Exploring Features**:
   - **My Predictions**: View your prediction history
   - **Model Test Results**: See detailed performance metrics for each model
   - **EDA Sections**: Explore data visualizations and distributions
   - **Model Description**: Learn about the algorithms used

## Project Structure

```
Multiple-Disease-Prediction/
├── app.py                          # Main application file
├── app_diabetes.py                 # Diabetes prediction module
├── app_heart.py                    # Heart disease prediction module
├── app_breast_cancer.py            # Breast cancer prediction module
├── user.py                         # User authentication and database functions
├── database.py                     # Database utilities
├── models/
│   ├── logistic_regression.py      # Logistic Regression implementation
│   ├── decision_tree.py            # Decision Tree implementation
│   └── svm.py                      # SVM implementation
├── utils/
│   ├── data_preprocessing.py       # Data preprocessing utilities
│   └── model_evaluation.py         # Model evaluation utilities
├── dataset/
│   ├── diabetes.csv                # Diabetes dataset
│   └── heart.csv                   # Heart disease dataset
├── EDA/                            # Exploratory Data Analysis images
│   ├── Heart_Disease_Pie.png
│   ├── Categorical_data.png
│   ├── Numerical_data.png
│   ├── Correlation_Heatmap.png
│   ├── pie-chart.png
│   ├── datadistribution_diabetes.png
│   └── heat-map.png
├── saved_models/                   # Trained model files
├── processed_data/                 # Preprocessed data files
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib & Seaborn**: Data visualization
- **SQLite**: Database for user authentication and prediction history
- **PIL**: Image processing

## Key Features Implementation

### Breast Cancer Prediction Specifics

1. **Feature Selection with PCA**:

   - Uses Principal Component Analysis to identify top 10 features from 30 original features
   - Reduces dimensionality while maintaining important information
   - Improves model performance and training speed

2. **SVM with Hinge Loss**:

   - Implements custom SVM with hinge loss function
   - Uses gradient descent for optimization
   - Includes L2 regularization to prevent overfitting

3. **Data Preprocessing**:

   - StandardScaler normalization for feature scaling
   - Label conversion from {0, 1} to {-1, +1} for SVM compatibility
   - Train-test split (80-20) for model evaluation

4. **Real-time Predictions**:
   - Input validation for all 10 features
   - Real-time prediction with probability output
   - Automatic saving to prediction history

## Model Performance

All models are evaluated on test datasets and provide:

- **Accuracy Scores**
- **Precision, Recall, and F1-Score**
- **Confusion Matrices**
- **Classification Reports**

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Heart Disease Dataset: Kaggle
- Diabetes Dataset: Publicly available medical datasets
- Breast Cancer Dataset: Wisconsin Breast Cancer Dataset (scikit-learn)
- Scikit-learn library for machine learning utilities

## Contact

For questions or issues, please open an issue on GitHub.
