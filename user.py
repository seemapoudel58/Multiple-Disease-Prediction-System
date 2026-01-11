import streamlit as st
import sqlite3
import re
import hashlib
from database import create_users_table 
import json
from datetime import datetime, timedelta


# Database connection function
def create_connection():
    return sqlite3.connect('new_user.db')

# Create Users Table
def create_users_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        email TEXT PRIMARY KEY,
                        username TEXT UNIQUE,
                        password TEXT)''')
    conn.commit()
    conn.close()

# Create Predictions Table
def create_predictions_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT,
                        disease TEXT,
                        input_parameters TEXT,
                        prediction_result TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Ensure both tables exist when the app starts
create_users_table()
create_predictions_table()

# Hashing function for passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Insert new user into the database
def insert_user(email, username, password):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)", (email, username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Fetch user details based on email
def get_user(email):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

# Fetch all registered emails
def get_user_emails():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users")
    emails = [row[0] for row in cursor.fetchall()]
    conn.close()
    return emails

# Fetch all registered usernames
def get_usernames():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users")
    usernames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return usernames

# Validate email format
def validate_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

# Validate username format
def validate_username(username):
    pattern = "^[a-zA-Z0-9]*$"
    return bool(re.match(pattern, username))

# Check if users table exists
def check_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
    table_exists = cursor.fetchone()
    conn.close()
    return bool(table_exists)

# Save user prediction to database
def save_user_prediction(email, disease, input_data, result):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_predictions (email, disease, input_parameters, prediction_result) VALUES (?, ?, ?, ?)", 
                   (email, disease, json.dumps(input_data), result))
    conn.commit()
    conn.close()

# Fetch user's past predictions
def get_user_predictions(email):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT disease, input_parameters, prediction_result, timestamp FROM user_predictions WHERE email=?", (email,))
    predictions = cursor.fetchall()
    conn.close()
    return predictions



# Sign Up function
def sign_up():
    """Sign up form"""
    with st.form(key='signup', clear_on_submit=True):
        st.subheader(':green[Sign Up]')
        email = st.text_input(':blue[Email]', placeholder='Enter Your Email')
        username = st.text_input(':blue[Username]', placeholder='Enter Your Username')
        password1 = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')
        password2 = st.text_input(':blue[Confirm Password]', placeholder='Confirm Your Password', type='password')

        if st.form_submit_button('Sign Up'):
            # Debugging output
            emails = get_user_emails()
            usernames = get_usernames()
            st.write("Emails in DB:", emails)
            st.write("Usernames in DB:", usernames)

            if not email or not username or not password1 or not password2:
                st.warning('All fields are required')
                return

            if not validate_email(email):
                st.warning('Invalid Email')
                return

            if email in emails:
                st.warning('Email Already Exists!!')
                return

            if not validate_username(username):
                st.warning('Invalid Username')
                return

            if username in usernames:
                st.warning('Username Already Exists')
                return

            if len(username) < 2:
                st.warning('Username Too Short')
                return

            if len(password1) < 6:
                st.warning('Password is too Short')
                return

            if password1 != password2:
                st.warning('Passwords Do Not Match')
                return

            # Hash password and store in database
            hashed_password = hash_password(password2)
            success = insert_user(email, username, hashed_password)
            if success:
                st.success('Account created successfully!!')
                st.balloons()
            else:
                st.error('Error creating account. Try again.')

def login():
    with st.form(key='login'):
        st.subheader(':green[Login]')
        email = st.text_input(':blue[Email]', placeholder='Enter Your Email')
        password = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')

        if st.form_submit_button('Login'):
            user = get_user(email)
            if user:
                stored_password = user[2]
                if hash_password(password) == stored_password:
                    st.success('Login successful!')
                    st.session_state.logged_in = True
                    st.session_state.user_email = email  # Store user email
                    st.rerun()
                else:
                    st.error('Incorrect password')
            else:
                st.error('User not found')


def check_recent_predictions(email, disease_type):
    conn = sqlite3.connect('new_user.db')
    cursor = conn.cursor()
    
    # Get the current date and the date 30 days ago
    current_date = datetime.now()
    thirty_days_ago = current_date - timedelta(days=30)
    
    # Define positive prediction messages
    positive_results = {
        "Heart Disease": "The person has heart disease",
        "Diabetes": 1
    }

    # Query to get all the positive predictions for the specific disease within the last 30 days
    cursor.execute("""
        SELECT * FROM user_predictions
        WHERE email = ? AND disease = ? AND prediction_result = ? AND timestamp >= ?
    """, (email, disease_type, positive_results[disease_type], thirty_days_ago.strftime('%Y-%m-%d %H:%M:%S')))
    
    predictions = cursor.fetchall() # this function used to retrieve all rows from result of an executed query
    conn.close()

    # Return True if there are 3 or more positive predictions in the last 30 days
    return len(predictions) >= 3





