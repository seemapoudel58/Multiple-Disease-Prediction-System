import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from utils.data_preprocessing import DataPreprocessor
from models.logistic_regression import LogisticRegression
from utils.model_evaluation import ModelEvaluator
from datetime import datetime, timedelta
from user import check_recent_predictions

# Load and preprocess data
preprocessor = DataPreprocessor("dataset/heart.csv")
preprocessor.load_data()
preprocessor.normalize_data()
X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data()
preprocessor.save_data(X_train, y_train, X_val, y_val, X_test, y_test)

# Train the model
model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
model.save_model()

def save_user_prediction(email, disease, input_data, result):
    conn = sqlite3.connect('new_user.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_predictions (email, disease, input_parameters, prediction_result) VALUES (?, ?, ?, ?)", 
                   (email, disease, json.dumps(input_data), result))
    conn.commit()
    conn.close()



def app_heartdisease(model):
    st.title('Heart Disease Prediction using ML')
    
    email = st.session_state.get('user_email', 'Guest')  # Get logged-in user's email
    
    # Input Fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, step=1)
    with col2:
        sex = 1 if st.selectbox('Sex', ['Male', 'Female']) == 'Male' else 0
    with col3:
        cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        cp = cp_options.index(st.selectbox('Chest Pain Type', cp_options))
    
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, step=1)
    with col2:
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)
    with col3:
        fbs = 1 if st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes']) == 'Yes' else 0
    
    with col1:
        restecg_options = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy']
        restecg = restecg_options.index(st.selectbox('Resting ECG Results', restecg_options))
    
    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, step=1)
    with col3:
        exang = 1 if st.selectbox('Exercise Induced Angina', ['No', 'Yes']) == 'Yes' else 0
    
    with col1:
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, step=0.1)
    with col2:
        slope_options = ['Upsloping', 'Flat', 'Downsloping']
        slope = slope_options.index(st.selectbox('Slope of Peak Exercise ST Segment', slope_options))
    
    with col3:
        ca = int(st.selectbox('Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3', '4']))
    
    with col1:
        thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
        thal = thal_options.index(st.selectbox('Thalassemia Type', thal_options))
    
    heart_diagnosis = ''
    show_performance = False
    

    # Check if user has predicted heart disease 3 times within the last 30 days
    if check_recent_predictions(email, 'Heart Disease'):
        st.warning("⚠️ **Alert:** You have predicted **Heart Disease** 3 or more times in the last 30 days with a positive result. Please consult a doctor.")

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_prediction = model.predict([user_input])
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        st.success(heart_diagnosis)
        show_performance = True
        
        # Save data to database
        save_user_prediction(email, "Heart Disease", user_input, heart_diagnosis)


def show_heart_model_test_result():
    """Display Heart Disease Model Test Results"""
    
    # Explanation about test data
    st.subheader("📊 Why is Test Data Important?")
    st.write(
        "Machine Learning models are trained on historical data, but we need to ensure that they "
        "generalize well to **new, unseen data**. That's why we split the dataset into **training, "
        "validation, and test sets.**"
    )
    
    # Display test data percentage
    test_ratio = 1 - (0.7 + 0.15)  # Assuming train_ratio=0.7, val_ratio=0.15
    st.info(f"🩺 **Test Data Percentage:** {test_ratio * 100:.2f}% of total data.")
    
    st.title("Heart Disease Model Test Results")
    
    # Display model evaluation images and descriptions in separate columns
    col1, col3 = st.columns([7, 3.27])

    with col1:
        # Model performance image
        st.subheader("Model Performance on Test Data")
        st.image('heart_disease_metrics_vertical.png', caption="Heart Disease Model Performance")
        st.write(
            "This figure shows the performance metrics of the Heart Disease model, including accuracy, precision, "
            "recall, and F1-score. These metrics help us assess how well the model predicts the presence or absence "
            "of heart disease."
        )
        
    with col3:
        # Confusion matrix image
        st.subheader("Confusion Matrix")
        st.image('heart_disease_confusion_matrix.png', caption="Confusion Matrix")
        st.write(
            "A confusion matrix helps us evaluate the performance of the classification model. "
            "It shows the counts of true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)."
        )
        st.write(
    "📌 **Key Terms in Confusion Matrix:**\n"
    "- **True Positives (TP):** Correctly predicted heart disease cases.\n"
    "- **False Positives (FP):** Incorrectly predicted heart disease cases (patients without heart disease predicted to have it).\n"
    "- **True Negatives (TN):** Correctly predicted non-heart disease cases.\n"
    "- **False Negatives (FN):** Incorrectly predicted non-heart disease cases (patients with heart disease predicted not to have it)."
)

def show_eda_for_heart_disease():
    """Displays EDA (Exploratory Data Analysis) Results for Heart Disease"""

    st.title("Exploratory Data Analysis (EDA) for Heart Disease")

    # Display Short Description
    st.subheader("📊 What is Exploratory Data Analysis (EDA)?")
    st.write(
        "EDA is an approach to analyzing datasets to summarize their main characteristics, "
        "often with visual methods. It helps us understand the structure of data, uncover "
        "patterns, detect anomalies, and test assumptions before proceeding with modeling."
    )

    st.write(
        "In this section, we present several visualizations that provide insights into the "
        "distribution of data for various heart disease parameters. These plots can help us "
        "understand the relationships between different features and heart disease diagnosis."
    )

    # Get the base directory relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eda_dir = os.path.join(base_dir, 'EDA')

    # Display Heart Disease Pie Chart (for diagnosis distribution)
    st.subheader("🫀 Distribution of Heart Disease Diagnoses (Pie Chart)")
    pie_chart_path = os.path.join(eda_dir, 'Heart_Disease_Pie.png')
    if os.path.exists(pie_chart_path):
        st.image(pie_chart_path, caption="Distribution of Heart Disease Diagnosis (1: Disease Present, 0: Disease Absent)", width=600)
    else:
        st.error(f"Image not found: {pie_chart_path}")
    st.write(
        "This pie chart illustrates the proportion of individuals diagnosed with heart disease "
        "versus those without it. A higher percentage of one category over the other may indicate "
        "an imbalance in the dataset that we need to account for when training the model."
    )

    # Display Categorical Data Distribution
    st.subheader("📊 Distribution of Categorical Data")
    categorical_path = os.path.join(eda_dir, 'Categorical_data.png')
    if os.path.exists(categorical_path):
        st.image(categorical_path, caption="Distribution of Categorical Features (e.g., Chest Pain Type, Fasting Blood Sugar)")
    else:
        st.error(f"Image not found: {categorical_path}")
    st.write(
        "This bar plot shows the distribution of categorical features, such as chest pain type and "
        "fasting blood sugar. These features are important for diagnosing heart disease and understanding "
        "how different categories (e.g., chest pain types) are distributed among the individuals."
    )

    # Display Numerical Data Distribution
    st.subheader("📈 Distribution of Numerical Data")
    numerical_path = os.path.join(eda_dir, 'Numerical_data.png')
    if os.path.exists(numerical_path):
        st.image(numerical_path, caption="Distribution of Numerical Features (e.g., Age, Cholesterol, Blood Pressure)")
    else:
        st.error(f"Image not found: {numerical_path}")
    st.write(
        "This plot shows the distribution of numerical features like age, cholesterol levels, and blood pressure. "
        "Understanding the distribution of these features helps in identifying patterns and outliers, which can influence the model's performance."
    )

    # Display Correlation Heatmap
    st.subheader("🔑 Correlation Heatmap of Features")
    heatmap_path = os.path.join(eda_dir, 'Correlation_Heatmap.png')
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, caption="Correlation Heatmap (Shows how features are related to each other)")
    else:
        st.error(f"Image not found: {heatmap_path}")
    st.write(
        "The correlation heatmap shows the strength of relationships between different features. "
        "It helps us identify which features are strongly correlated with heart disease diagnosis and with each other. "
        "For example, a high correlation between cholesterol levels and heart disease may suggest that this feature is an important predictor."
    )



def show_logistic_regression_description():
    """Display Logistic Regression Model Description for Heart Disease Prediction"""

    st.title("Model Description: Logistic Regression for Heart Disease Prediction")

    # Logistic Regression Explanation
    st.subheader("📊 Logistic Regression Overview")
    st.write("""
    Logistic Regression is a type of statistical model used for binary classification tasks, 
    such as predicting whether a person has heart disease or not. It is based on the following equation:
    
    $$y = \sigma(wX + b)$$

    Where:
    - $y$ is the predicted probability of the outcome (i.e., heart disease).
    - $w$ represents the weights of the features.
    - $X$ is the feature vector (i.e., the inputs like age, sex, blood pressure, etc.).
    - $b$ is the bias term.
    - $\sigma$ is the **sigmoid function** that maps the linear combination of features to a probability value between 0 and 1.
    """)

    # Sigmoid Function Explanation
    st.subheader("🔑 Sigmoid Function")
    st.write("""
    The **sigmoid function** is crucial in logistic regression as it transforms the output of the linear equation 
    into a probability. 

    The sigmoid function is mathematically defined as:

    $$\sigma(z) = \\frac{1}{1 + e^{-z}}$$

    - It ensures that the output is between **0 and 1**, making it suitable for probability estimation.
    **.
    """)

    # Generate and show Sigmoid Function Diagram
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y)
    ax.set_title("Sigmoid Function")
    ax.set_xlabel("z")
    ax.set_ylabel("σ(z)")
    plt.savefig("sigmoid_function_resized.png", dpi=100, bbox_inches='tight')
    st.image("sigmoid_function_resized.png", caption="Sigmoid Function")

    # Binary Cross-Entropy (BCE) Loss Explanation
    st.subheader("⚖️ Binary Cross-Entropy Loss (BCE Loss)")
    st.write("""
    **Binary Cross-Entropy Loss (BCE Loss)** is used as the cost function in logistic regression for binary classification.
    It measures how well the model's predicted probabilities match the true labels. 

    The **BCE loss formula** is:

    $$L(y, \hat{y}) = - \\left[ y \\log(\\hat{y}) + (1 - y) \\log(1 - \\hat{y}) \\right]$$

    Where:
    - **$y$** is the actual class label (0 or 1).
    - **$\hat{y}$** is the predicted probability from the logistic regression model.
    - The loss is minimized when the predicted probability is close to the actual class label.
    """)
    # Load and resize BCE loss diagram
    image = Image.open("bce_loss.png").resize((300, 200))
    st.image(image, caption="Binary Cross-Entropy Loss")


        # Gradient Descent Explanation
    st.subheader("🔄 Gradient Descent")
    st.write("""
    **Gradient Descent** is an optimization algorithm used to minimize the loss function by iteratively adjusting the model's 
    parameters (weights and bias). It updates the model's parameters in the direction of the steepest decrease in the 
    loss function, as computed by the gradient.

    The weight update formula in gradient descent is:

    $$ w^{(t+1)} = w^{(t)} - \\alpha \\frac{\partial L}{\partial w} $$

    Where:
    - **$w^{(t+1)}$** is the updated weight after each iteration.
    - **$w^{(t)}$** is the current weight value.
    - **$\\alpha$** is the learning rate, which controls how big the step is.
    - **$\\frac{\partial L}{\partial w}$** is the gradient of the loss function with respect to the weight.
    - The process continues until the loss converges to a minimum.
    """)

    # Load and resize gradient descent diagram
    image = Image.open("gradient_descent.png").resize((400, 200))
    st.image(image, caption="Gradient Descent")

    # Why Logistic Regression for Heart Disease Prediction?
    st.subheader("💡 Why Logistic Regression for Heart Disease Prediction?")
    st.write("""
    Logistic regression is an ideal choice for heart disease prediction because:
    - **Binary Classification**: The problem of predicting whether a person has heart disease or not is a binary classification task, 
      which aligns perfectly with logistic regression.
    - **Interpretable**: Logistic regression is a simple and interpretable model, making it easier to understand how the 
      input features (e.g., age, cholesterol, blood pressure) influence the prediction.
    - **Efficient**: Logistic regression performs well even with smaller datasets and is computationally efficient.
    - **Probabilistic Output**: The model provides probabilities, which allows us to measure the confidence of the prediction 
      (e.g., 80% chance of having heart disease).

    These characteristics make logistic regression a reliable model for predicting heart disease.
    """)

