import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import sqlite3
import json
import os
from models.svm import SVM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def save_user_prediction(email, disease, input_data, result):
    """Save user prediction to database"""
    conn = sqlite3.connect('new_user.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_predictions (email, disease, input_parameters, prediction_result) VALUES (?, ?, ?, ?)", 
                   (email, disease, json.dumps(input_data), result))
    conn.commit()
    conn.close()

def app_breast_cancer():

    # Load dataset from sklearn
    @st.cache_data
    def load_data():
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = data.target  # Add target column (0 = Benign, 1 = Malignant)
        return df, data

    # Function to get the top features based on PCA
    def get_top_pca_features(df, n_components=10):
        X = df.drop(columns=["diagnosis"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA with all components
        pca = PCA()
        pca.fit(X_scaled)

        # Get absolute loadings for features
        loadings = abs(pca.components_)
        loadings_df = pd.DataFrame(loadings, columns=df.columns[:-1])

        # Transpose for easier access
        loadings_df = loadings_df.T
        loadings_df.columns = [f'PC{i+1}' for i in range(loadings_df.shape[1])]

        # Extract top 10 unique features
        top_10_features = []
        for i in range(n_components):
            component_loadings = loadings_df.sort_values(by=f'PC{i+1}', ascending=False)
            top_10_features.extend(component_loadings.index[:10].tolist())

        # Remove duplicates while preserving order
        unique_top_10_features = []
        for feature in top_10_features:
            if feature not in unique_top_10_features:
                unique_top_10_features.append(feature)

        return unique_top_10_features[:10], scaler, pca

    # Load and preprocess the data
    df, data_info = load_data()
    top_features, scaler, pca = get_top_pca_features(df)

    # Split data for training and testing
    def preprocess_data(df,selected_features, n_components=10):
        X = df[selected_features]
        y = df["diagnosis"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        return train_test_split(X_pca, y, test_size=0.2, random_state=42), scaler, pca

    (X_train, X_test, y_train, y_test), scaler, pca = preprocess_data(df,selected_features=top_features, n_components=10)

    # Train the SVM model
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=2000)
    svm.fit(X_train, y_train)

    st.title('Breast Cancer Prediction using ML')
    
    email = st.session_state.get('user_email', 'Guest')  # Get logged-in user's email

    # Collect user inputs
    st.subheader("🔹 Input Patient Details")
    user_input = {}
    cols = st.columns(3)

    for i, col_name in enumerate(top_features):
        with cols[i % 3]:
            user_input[col_name] = st.number_input(
                label=col_name.replace("_", " ").capitalize(),
                # value=None,
                value=float(df[col_name].mean()),
                step=0.0001,
                format="%.6f"
            )

    # Prediction Button
    if st.button('Breast Cancer Test Result'):
        if any(value == None for value in user_input.values()):
            st.warning("Please fill in all input fields.")
        else:
            user_input_array = np.array([list(user_input.values())]).reshape(1, -1)
            user_input_scaled = scaler.transform(user_input_array)
            
            bc_prediction = svm.predict(user_input_scaled)

            diagnosis = 'Malignant (Cancerous)' if bc_prediction[0] == 1 else 'Benign (Non-Cancerous)'
            st.success(f'Prediction: {diagnosis}')
            
            # Save prediction to database
            save_user_prediction(email, "Breast Cancer", user_input, diagnosis)

            # Model Performance Section
            st.subheader(" Model Performance on Test Data")
            y_pred = svm.predict(X_test)
            y_pred = np.where(y_pred == 1, 1, 0)

            # Accuracy Score
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Model Accuracy:** {accuracy:.2f}")

            # Classification Report
            st.write(" **Classification Report:**")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)
            st.dataframe(report_df)

            # Correlation Matrix Section
            st.subheader("Correlation Matrix")
            corr_matrix = df[top_features].corr()

            # Plot the heatmap of the correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title("Correlation Matrix")
            st.pyplot(fig)

            # Confusion Matrix
            st.write(" **Confusion Matrix:**")
            conf_matrix = confusion_matrix(y_test, y_pred)
        
            # Check for unique labels in y_test
            unique_labels = np.unique(y_test)
        
            # Plot the heatmap with dynamic tick labels
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_labels, yticklabels=unique_labels)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

def show_breast_cancer_model_test_result():
    """Display Breast Cancer Model Test Results"""
    
    # Explanation about test data
    st.subheader("📊 Why is Test Data Important?")
    st.write(
        "Machine Learning models are trained on historical data, but we need to ensure that they "
        "generalize well to **new, unseen data**. That's why we split the dataset into **training, "
        "validation, and test sets.**"
    )
    
    # Display test data percentage
    test_ratio = 0.2  # 20% test data based on train_test_split
    st.info(f"🩺 **Test Data Percentage:** {test_ratio * 100:.2f}% of total data.")
    
    st.title("Breast Cancer Model Test Results")
    
    # Load data and train model to get test results
    @st.cache_data
    def load_data():
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = data.target
        return df, data
    
    def get_top_pca_features(df, n_components=10):
        X = df.drop(columns=["diagnosis"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        loadings = abs(pca.components_)
        loadings_df = pd.DataFrame(loadings, columns=df.columns[:-1])
        loadings_df = loadings_df.T
        loadings_df.columns = [f'PC{i+1}' for i in range(loadings_df.shape[1])]
        top_10_features = []
        for i in range(n_components):
            component_loadings = loadings_df.sort_values(by=f'PC{i+1}', ascending=False)
            top_10_features.extend(component_loadings.index[:10].tolist())
        unique_top_10_features = []
        for feature in top_10_features:
            if feature not in unique_top_10_features:
                unique_top_10_features.append(feature)
        return unique_top_10_features[:10], scaler, pca
    
    def preprocess_data(df, selected_features, n_components=10):
        X = df[selected_features]
        y = df["diagnosis"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        return train_test_split(X_pca, y, test_size=0.2, random_state=42), scaler, pca
    
    df, data_info = load_data()
    top_features, _, _ = get_top_pca_features(df)
    
    # Train model and get test data in one split
    (X_train, X_test, y_train, y_test), scaler, pca = preprocess_data(df, selected_features=top_features, n_components=10)
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=2000)
    svm.fit(X_train, y_train)
    
    # Get predictions
    y_pred = svm.predict(X_test)
    y_pred = np.where(y_pred == 1, 1, 0)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Display metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Performance Metrics")
        st.write(f"**Accuracy:** {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification Report
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(4)
        st.dataframe(report_df)
        
        st.write(
            "This figure shows the performance metrics of the Breast Cancer model, including accuracy, precision, "
            "recall, and F1-score. These metrics help us assess how well the model predicts whether a tumor is "
            "malignant (cancerous) or benign (non-cancerous)."
        )
    
    with col2:
        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        unique_labels = np.unique(y_test)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Benign', 'Malignant'], 
                    yticklabels=['Benign', 'Malignant'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        
        st.write(
            "A confusion matrix helps us evaluate the performance of the classification model. "
            "It shows the counts of true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)."
        )
        st.write(
            "📌 **Key Terms in Confusion Matrix:**\n"
            "- **True Positives (TP):** Correctly predicted malignant cases.\n"
            "- **False Positives (FP):** Incorrectly predicted malignant cases (benign tumors predicted as malignant).\n"
            "- **True Negatives (TN):** Correctly predicted benign cases.\n"
            "- **False Negatives (FN):** Incorrectly predicted benign cases (malignant tumors predicted as benign)."
        )

def show_eda_for_breast_cancer():
    """Displays EDA (Exploratory Data Analysis) Results for Breast Cancer"""

    st.title("Exploratory Data Analysis (EDA) for Breast Cancer")

    # Display Short Description
    st.subheader("📊 What is Exploratory Data Analysis (EDA)?")
    st.write(
        "EDA is an approach to analyzing datasets to summarize their main characteristics, "
        "often with visual methods. It helps us understand the structure of data, uncover "
        "patterns, detect anomalies, and test assumptions before proceeding with modeling."
    )

    st.write(
        "In this section, we present several visualizations that provide insights into the "
        "distribution of data for various breast cancer parameters. These plots can help us "
        "understand the relationships between different features and breast cancer diagnosis."
    )

    # Get the base directory relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eda_dir = os.path.join(base_dir, 'EDA')

    # Load data for generating visualizations
    @st.cache_data
    def load_data():
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = data.target  # 0 = Benign, 1 = Malignant
        return df, data

    df, data_info = load_data()

    # Display Breast Cancer Diagnosis Distribution (Pie Chart)
    st.subheader("🫀 Distribution of Breast Cancer Diagnoses (Pie Chart)")
    
    # Create pie chart on the fly
    diagnosis_counts = df['diagnosis'].value_counts()
    labels = ['Benign (Non-Cancerous)', 'Malignant (Cancerous)']
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(diagnosis_counts, labels=labels, autopct='%1.1f%%', colors=colors, 
           startangle=140, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    ax.set_title('Breast Cancer Diagnosis Distribution', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    
    st.write(
        "This pie chart illustrates the proportion of tumors diagnosed as malignant (cancerous) "
        "versus benign (non-cancerous). Understanding this distribution helps us assess the balance "
        "of the dataset and the model's ability to predict both classes effectively."
    )

    # Display Distribution of Numerical Features
    st.subheader("📈 Distribution of Numerical Features")
    
    # Select a few key features to display
    key_features = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness']
    available_features = [f for f in key_features if f in df.columns]
    
    if available_features:
        num_cols = len(available_features)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:6]):  # Show up to 6 features
            sns.histplot(df[feature], kde=True, bins=20, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature.title()}')
            axes[i].set_xlabel(feature.title())
            axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(available_features), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write(
            "This plot shows the distribution of key numerical features like mean radius, mean texture, "
            "mean area, and mean perimeter. Understanding the distribution of these features helps in "
            "identifying patterns, outliers, and the overall characteristics of the dataset."
        )

    # Display Correlation Heatmap
    st.subheader("🔑 Correlation Heatmap of Features")
    
    # Use top features for correlation heatmap
    top_features = df.columns[:-1][:15]  # Use first 15 features for readability
    corr_matrix = df[top_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    plt.title("Correlation Heatmap of Breast Cancer Features")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write(
        "The correlation heatmap shows the strength of relationships between different features. "
        "It helps us identify which features are strongly correlated with breast cancer diagnosis "
        "and with each other. For example, features like mean radius, mean perimeter, and mean area "
        "are often highly correlated, which is expected as they measure related tumor characteristics."
    )

def show_svm_description():
    """Display Support Vector Machine (SVM) Model Description for Breast Cancer Prediction"""

    st.title("Model Description: Support Vector Machine (SVM) for Breast Cancer Prediction")

    # SVM Explanation
    st.subheader("📊 Support Vector Machine (SVM) Overview")
    st.write("""
    A **Support Vector Machine (SVM)** is a powerful supervised learning algorithm used for classification and regression tasks. 
    SVM works by finding the optimal hyperplane that best separates different classes in the feature space. The goal is to maximize 
    the margin between the hyperplane and the nearest data points (support vectors) from each class.
    
    For binary classification tasks like breast cancer prediction, SVM aims to:
    - Find the best decision boundary (hyperplane) that separates malignant (cancerous) tumors from benign (non-cancerous) tumors.
    - Maximize the margin between the hyperplane and the support vectors (the data points closest to the decision boundary).
    - Use the hyperplane to classify new data points based on which side of the boundary they fall on.
    
    The mathematical equation for the decision function is:
    
    $$f(x) = w^T x - b$$
    
    Where:
    - $w$ is the weight vector (normal to the hyperplane).
    - $x$ is the input feature vector.
    - $b$ is the bias term.
    - If $f(x) \geq 0$, the prediction is one class (e.g., malignant).
    - If $f(x) < 0$, the prediction is the other class (e.g., benign).
    """)

    # Hyperplane and Margin Explanation
    st.subheader("🔑 Hyperplane and Margin")
    st.write("""
    The **hyperplane** is the decision boundary that separates the classes. In a 2D space, this is a line; in 3D, it's a plane; 
    and in higher dimensions, it's a hyperplane.
    
    The **margin** is the distance between the hyperplane and the nearest data points from each class. SVM aims to maximize this margin 
    to create the most robust decision boundary. The data points that lie on the margin are called **support vectors**.
    
    The margin width is calculated as:
    
    $$Margin = \\frac{2}{||w||}$$
    
    Maximizing the margin is equivalent to minimizing $||w||^2$, subject to the constraint that all data points are correctly classified 
    with a margin of at least 1.
    """)

    # Hinge Loss Explanation
    st.subheader("⚖️ Hinge Loss Function")
    st.write("""
    SVM uses the **Hinge Loss** function, which is designed to penalize misclassifications and points that are too close to the decision boundary.
    
    The hinge loss function is defined as:
    
    $$L(y, f(x)) = \\max(0, 1 - y \\cdot f(x))$$
    
    Where:
    - $y$ is the actual class label (converted to -1 or +1 for SVM).
    - $f(x) = w^T x - b$ is the decision function output.
    
    The loss is:
    - **0** when $y \\cdot f(x) \geq 1$ (correctly classified with sufficient margin).
    - **$1 - y \\cdot f(x)$** when $y \\cdot f(x) < 1$ (misclassified or too close to the boundary).
    
    The total loss function with regularization is:
    
    $$L = \\frac{1}{n} \\sum_{i=1}^{n} \\max(0, 1 - y_i (w^T x_i - b)) + \\lambda ||w||^2$$
    
    Where:
    - $\\lambda$ is the regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors.
    - The $||w||^2$ term prevents overfitting by penalizing large weights.
    """)

    # Gradient Descent for SVM
    st.subheader("🔄 Gradient Descent Optimization")
    st.write("""
    SVM uses **Gradient Descent** to minimize the loss function and find the optimal weights and bias.
    
    The weight update formula in gradient descent for SVM is:
    
    $$w^{(t+1)} = w^{(t)} - \\alpha \\left( 2\\lambda w^{(t)} - x_i y_i \\right)$$
    
    $$b^{(t+1)} = b^{(t)} - \\alpha (-y_i)$$
    
    When $y_i (w^T x_i - b) < 1$ (misclassification or insufficient margin), otherwise:
    
    $$w^{(t+1)} = w^{(t)} - \\alpha (2\\lambda w^{(t)})$$
    
    Where:
    - **$w^{(t+1)}$** is the updated weight after each iteration.
    - **$w^{(t)}$** is the current weight value.
    - **$\\alpha$** is the learning rate, which controls how big the step is.
    - **$\\lambda$** is the regularization parameter.
    - The process continues iteratively until convergence or a maximum number of iterations is reached.
    """)

    # Label Conversion
    st.subheader("🔄 Label Conversion")
    st.write("""
    SVM requires labels to be in the format **{-1, +1}** instead of **{0, 1}** for the hinge loss function to work correctly.
    
    The conversion is:
    - Class 0 (Benign) → -1
    - Class 1 (Malignant) → +1
    
    After prediction, the output is converted back:
    - If $f(x) \geq 0$ → Predict as +1 → Class 1 (Malignant)
    - If $f(x) < 0$ → Predict as -1 → Class 0 (Benign)
    """)

    # Why SVM for Breast Cancer Prediction?
    st.subheader("💡 Why Support Vector Machine (SVM) for Breast Cancer Prediction?")
    st.write("""
    SVM is an excellent choice for breast cancer prediction because:
    
    - **Effective for High-Dimensional Data**: Breast cancer datasets often have many features (30 features in the Wisconsin dataset). 
      SVM performs well even when the number of features is large compared to the number of samples.
    
    - **Robust Decision Boundary**: By maximizing the margin, SVM creates a robust decision boundary that is less sensitive to 
      outliers and new data points, making it more reliable for medical diagnosis.
    
    - **Memory Efficient**: SVM only uses support vectors (the data points on or near the margin) for prediction, making it memory 
      efficient. Once trained, only the support vectors are needed for classification.
    
    - **Works Well with PCA**: SVM works effectively with dimensionality reduction techniques like Principal Component Analysis (PCA), 
      which is used in the breast cancer prediction system to reduce feature complexity while maintaining important information.
    
    - **Handles Non-Linear Relationships**: While the basic SVM uses a linear kernel, it can be extended with kernel functions to 
      capture non-linear relationships between features, though the current implementation uses a linear kernel.
    
    - **Good Generalization**: The regularization term ($\\lambda ||w||^2$) helps prevent overfitting, ensuring the model generalizes 
      well to new, unseen breast cancer cases.
    
    These characteristics make SVM a reliable and effective model for predicting breast cancer based on tumor characteristics.
    """)

# Allow running the app standalone
if __name__ == "__main__":
    app_breast_cancer()