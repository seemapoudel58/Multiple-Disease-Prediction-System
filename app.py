import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from streamlit_option_menu import option_menu
from src.auth.user import login, sign_up, get_user_predictions
from src.app.pages.diabetes_page import app_diabetes, show_diabetes_model_test_result, show_eda_for_diabetes, show_decision_tree_description
from src.app.pages.heart_page import app_heartdisease, model, show_heart_model_test_result, show_eda_for_heart_disease, show_logistic_regression_description
from src.app.pages.breast_cancer_page import app_breast_cancer, show_breast_cancer_model_test_result, show_eda_for_breast_cancer, show_svm_description

# Set page config at the top before any other Streamlit command
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def load_health_assistant():
    """Health Assistant Page"""
    email = st.session_state.get('user_email', 'Guest')  # Get logged-in user's email
    
    with st.sidebar:
      selected = option_menu(
        'Multiple Disease Prediction System',
        [
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'Breast Cancer Prediction',
            'My Predictions',
            'Heart Model Test Result',
            'Diabetes Model Test Result',
            'Breast Cancer Model Test Result',
            'EDA for Heart Disease',
            'EDA for Diabetes',
            'EDA for Breast Cancer',
            'Model Description'
        ],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'clock', 'bar-chart', 'bar-chart', 'bar-chart', 'graph-up', 'graph-up', 'graph-up', 'database'],
        default_index=1
    )
     

    if selected == 'Diabetes Prediction':
        app_diabetes()
    elif selected == 'Heart Disease Prediction':
        app_heartdisease(model)
    elif selected == 'Breast Cancer Prediction':
        app_breast_cancer()
    elif selected == 'My Predictions':
        st.subheader("📜 My Previous Predictions")
        predictions = get_user_predictions(email)
        if predictions:
            for disease, inputs, result, timestamp in predictions:
                st.write(f"🦠 **Disease:** {disease}")
                st.write(f"🕒 **Date:** {timestamp}")
                st.write(f"📊 **Inputs:** {inputs}")
                st.write(f"🖍 **Prediction:** {result}")
                st.markdown("---")
        else:
            st.info("No past predictions found.")
    elif selected == 'Heart Model Test Result':
        show_heart_model_test_result()
    elif selected == 'Diabetes Model Test Result':
        show_diabetes_model_test_result()
    elif selected == 'Breast Cancer Model Test Result':
        show_breast_cancer_model_test_result()
    elif selected ==   'EDA for Heart Disease':
        show_eda_for_heart_disease()
    elif selected ==   'EDA for Diabetes':
        show_eda_for_diabetes()
    elif selected ==   'EDA for Breast Cancer':
        show_eda_for_breast_cancer()
    elif selected ==   'Model Description':
        show_logistic_regression_description()
        show_decision_tree_description()
        show_svm_description()


def main():
    """Main Function to Display SignUp/Login"""
    if not st.session_state.logged_in:
        st.title("Welcome to Multiple Disease Prediction System 🏥")
        
        choice = st.radio("Choose an option", ['Sign Up', 'Login'])

        if choice == 'Sign Up':
            sign_up()
        elif choice == 'Login':
            login()
    else:
        load_health_assistant()

if __name__ == "__main__":
    main()
