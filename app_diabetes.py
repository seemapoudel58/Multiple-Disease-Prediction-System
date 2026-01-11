import streamlit as st
import sqlite3
import pickle
import json
from user import check_recent_predictions

# Function to save predictions in the database
def save_user_prediction(email, disease, input_data, result):
    conn = sqlite3.connect('new_user.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_predictions (email, disease, input_parameters, prediction_result) VALUES (?, ?, ?, ?)", 
                   (email, disease, json.dumps(input_data), result))
    conn.commit()
    conn.close()

# Validation function
def validation(user_input):
    return all([
        0 <= user_input[0] <= 20,
        0 <= user_input[1] <= 200,
        0 <= user_input[2] <= 140,
        0 <= user_input[3] <= 100,
        0 <= user_input[4] <= 800,
        0.00 <= user_input[5] <= 70.00,
        0.000 <= user_input[6] <= 3.000,
        0 <= user_input[7] <= 100
    ])

# Diabetes Prediction App
def app_diabetes():
    st.title('Diabetes Prediction using ML')

    if 'input_valid' not in st.session_state:
        st.session_state.input_valid = True

    email = st.session_state.get('user_email', 'Guest')  # Get logged-in user's email

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', value=2)
    with col2:
        Glucose = st.number_input('Glucose Level', value=100)
    with col3:
        BloodPressure = st.number_input('Blood Pressure', value=70)
    with col1:
        SkinThickness = st.number_input('Skin Thickness', value=35)
    with col2:
        Insulin = st.number_input('Insulin Level', value=100)
    with col3:
        BMI = st.number_input('BMI', value=32.00)
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', value=0.25, step=0.001, format="%.3f")
    with col2:
        Age = st.number_input('Age', value=25)

    # Load trained diabetes model
    diabetes_model = pickle.load(open('saved_models/diabetes_model_decision_tree.sav', 'rb'))

    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                  BMI, DiabetesPedigreeFunction, Age]

    # Validate input
    st.session_state.input_valid = validation(user_input)
    if not st.session_state.input_valid:
        st.error('Please ensure all values are within their valid ranges before proceeding')

    if st.button('Diabetes Test Result', disabled=not st.session_state.input_valid):
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])

        diab_diagnosis = (
            'The person **is diabetic** using Decision Tree Model.'
            if diab_prediction[0] == 1
            else 'The person **is not diabetic** using Decision Tree Model.'
        )
        st.success(diab_diagnosis)
          # Save prediction result in the database
        save_user_prediction(email, "Diabetes", user_input, int(diab_prediction[0]))
         # Check if user has predicted Diabetes 3 times within the last 30 days
    if check_recent_predictions(email, 'Diabetes'):
        st.warning("⚠️ **Alert:** You have predicted **Diabetes** 3 or more times in the last 30 days with a positive result. Please consult a doctor.")



def show_diabetes_model_test_result():
    """Display Diabetes Model Test Results"""
    
    # Explanation about test data
    st.subheader("📊 Why is Test Data Important?")
    st.write(
        "Machine Learning models are trained on historical data, but we need to ensure that they "
        "generalize well to **new, unseen data**. That's why we split the dataset into **training, "
        "validation, and test sets.**"
    )
    
    # Display test data percentage
    test_ratio = 0.20  # Assuming train_ratio=0.80 and test_ratio=0.20
    st.info(f"🩺 **Test Data Percentage:** {test_ratio * 100:.2f}% of total data.")
    
    st.title("Diabetes Model Test Results")
    
    # Display model evaluation images and descriptions in separate columns
    col1, col3 = st.columns([6, 3.27])

    with col1:
        # Model performance image
        st.subheader("Model Performance on Test Data")
        st.image('metrics_plot.png', caption="Diabetes Model Performance")
        st.write(
            "This figure shows the performance metrics of the Diabetes model, including accuracy, precision, "
            "recall, and F1-score. These metrics help us assess how well the model predicts the presence or absence "
            "of diabetes."
        )
        
    with col3:
        # Confusion matrix image
        st.subheader("Confusion Matrix")
        st.image('confusion_matrix_custom.png', caption="Confusion Matrix")
        st.write(
            "A confusion matrix helps us evaluate the performance of the classification model. "
            "It shows the counts of true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)."
        )
        st.write(
    "📌 **Key Terms in Confusion Matrix:**\n"
    "- **True Positives (TP):** Correctly predicted diabetes cases.\n"
    "- **False Positives (FP):** Incorrectly predicted diabetes cases (patients without diabetes predicted to have it).\n"
    "- **True Negatives (TN):** Correctly predicted non-diabetes cases.\n"
    "- **False Negatives (FN):** Incorrectly predicted non-diabetes cases (patients with diabetes predicted not to have it)."
)


def show_eda_for_diabetes():
    """Displays EDA (Exploratory Data Analysis) Results for Diabetes"""

    st.title("Exploratory Data Analysis (EDA) for Diabetes")

    # Display Short Description
    st.subheader("📊 What is Exploratory Data Analysis (EDA)?")
    st.write(
        "EDA is an approach to analyzing datasets to summarize their main characteristics, "
        "often with visual methods. It helps us understand the structure of data, uncover "
        "patterns, detect anomalies, and test assumptions before proceeding with modeling."
    )

    st.write(
        "In this section, we present several visualizations that provide insights into the "
        "distribution of data for various diabetes-related parameters. These plots can help us "
        "understand the relationships between different features and the diabetes diagnosis."
    )

    # Display Diabetes Pie Chart (for diagnosis distribution)
    st.subheader(" Distribution of Diabetes Diagnoses (Pie Chart)")
    st.image('./EDA/pie-chart.png', caption="Distribution of Diabetes Diagnosis" ,width=500)
    st.write(
        "This pie chart illustrates the proportion of individuals diagnosed with diabetes "
        "versus those without it. In our dataset, approximately **65.1%** of individuals do not have diabetes, "
        "while the remaining **34.9%** are diagnosed with diabetes. This imbalance in the dataset is an important consideration "
        "when training the model, as it may affect the model's ability to generalize and predict effectively for both classes."
    )

    # Display  Data Distribution
    st.subheader("📈 Distribution of  Data Among Multiple Features")
    st.image('./EDA/datadistribution_diabetes.png', caption="Distribution of Numerical Features (e.g., Age, Glucose, BMI, etc.)", )
    st.write(
        "This plot shows the distribution of numerical features like age, glucose levels, BMI, and others. "
        "Understanding the distribution of these features helps in identifying patterns and outliers, which can influence the model's performance."
    )

    # Display Correlation Heatmap
    st.subheader("🔑 Correlation Heatmap of Features")
    st.image('./EDA/heat-map.png', caption="Correlation Heatmap (Shows how features are related to each other)")
    st.write(
        "The correlation heatmap shows the strength of relationships between different features. "
        "It helps us identify which features are strongly correlated with diabetes diagnosis and with each other. "
        "For example, a high correlation between BMI and diabetes may suggest that this feature is an important predictor."
    )

def show_decision_tree_description():
    """Display Decision Tree Model Description for Diabetes Prediction"""

    st.title("Model Description: Decision Tree for Diabetes Prediction")

    # Decision Tree Explanation
    st.subheader("📊 Decision Tree Overview")
    st.write("""
    A **Decision Tree** is a non-linear model used for both classification and regression tasks. It works by 
    recursively splitting the data based on feature values to form a tree-like structure. Each node in the tree represents a 
    decision based on a feature, and each leaf node represents the predicted class (i.e., heart disease presence or absence).

    The decision tree works by:
    - Splitting the dataset into subsets based on the feature values that provide the most information gain (or reduce entropy).
    - Continuing the process until all the data points in each subset are classified or a stopping criterion is met.
    
    The final tree structure helps make predictions by traversing down the tree, from the root node to a leaf node.
    """)

    # Decision Tree Diagram
    st.subheader("🌳 Decision Tree Structure")
    st.markdown("""
    The tree structure can be visualized as a series of decisions based on feature values (e.g. insulin, glucose, BMI, etc.).
    At each node, a decision is made to split the data, and the tree grows until each path leads to a final classification (diabetes or not).

    Here’s an example diagram of a decision tree for diabetes prediction:
    """)
    # Show Decision Tree Diagram
    st.image('decision_tree_example.png', caption="Example Decision Tree for Diabetes Prediction", width=800)

    # Gini Impurity and Information Gain
    st.markdown("""
    **Gini Impurity** and **Information Gain** are key concepts in Decision Trees.

    ### Gini Impurity:
    Decision Trees use **Gini Impurity** or **Entropy** to measure the "impurity" of a node, which helps in deciding the best split.

    The **Gini Impurity** for a node is calculated as:

    $$Gini = 1 - ∑(p_i^2)$$

    Where:
    - $p_i$ is the proportion of class $i$ in the node.

    The decision tree aims to minimize Gini Impurity at each node by choosing the feature and threshold that splits the data into the most pure subsets.
    """)

    st.markdown("""
    ### Information Gain:
    **Information Gain** is another way to measure the effectiveness of a split. It is based on the reduction of entropy:

    """)
    # Using st.latex() to render the formula
    st.latex(r'''
    Information\ Gain = Gini (Parent) - \sum \left(\frac{|Subset|}{|Parent|} \times Gini (Subset)\right)
    ''')

    st.write("""
    This helps to find the most informative features for splitting the data.
    """)

    # Show Gini and Information Gain Diagrams
    # st.image('gini_impurity.png', caption="Gini Impurity", use_column_width=True)
    # st.image('information_gain.png', caption="Information Gain", use_column_width=True)

    # Overfitting and Pruning
    st.subheader("⚖️ Overfitting and Pruning")
    st.write("""
    One challenge with decision trees is **overfitting**, where the tree becomes too complex and starts to model noise in the data.
    
    **Pruning** is the process of cutting back the tree by removing branches that provide little predictive value. This helps:
    - Prevent overfitting.
    - Improve the generalization of the model.
    - Simplify the model, making it more interpretable.
    
    Pruning can be done through various strategies like pre-pruning (limiting the tree depth) and post-pruning (removing nodes after the tree is fully grown).
    """)

    # Show Overfitting and Pruning Diagrams
   

    # Why Decision Tree for Heart Disease Prediction?
    st.subheader("💡 Why Decision Tree for Diabetes Prediction?")
    st.write("""
    Decision Trees are an ideal choice for diabetes prediction because:
- **Non-Linear Relationships**: Decision Trees can capture non-linear relationships between the features (e.g., glucose levels, insulin, BMI, etc.), which may not be effectively modeled by linear models like logistic regression.
- **Interpretability**: The tree structure provides an intuitive and interpretable model. You can easily visualize how decisions are made based on the different features, which is crucial in healthcare applications where interpretability is important.
- **No Feature Scaling Needed**: Unlike other models such as SVMs or KNN, Decision Trees do not require normalization or scaling of features. This simplifies the preprocessing steps.
- **Handling Missing Data**: Decision Trees can handle missing values effectively by choosing the best split based on available data, making it a robust model in real-world scenarios where data may be incomplete.
- **Easy to Implement**: Decision Trees are relatively simple to implement and computationally efficient, making them a good choice for large-scale datasets.
    """)
