import sqlite3

# Connect to the SQLite database
def get_db_connection():
    conn = sqlite3.connect('your_database.db')  # Adjust the database path if necessary
    conn.row_factory = sqlite3.Row  # Allows access to columns by name
    return conn

# Function to create the users table if it doesn't exist
def create_users_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        username TEXT NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# Function to insert a user into the database
def insert_user(email, username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)", (email, username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:  # For handling duplicate entries
        conn.close()
        return False

# Function to get a user by email
def get_user(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

# Function to get all user emails
def get_user_emails():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users")
    emails = [row[0] for row in cursor.fetchall()]
    conn.close()
    return emails

# Function to get all usernames
def get_usernames():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users")
    usernames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return usernames

# Call this function to ensure the users table exists when the app starts
create_users_table()
