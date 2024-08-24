# Importing the dependencies
import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import os
import bcrypt
from PIL import Image
import io

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Universal Styles */
    body, h1, h2, h3, p, div {
        font-family: 'Arial', sans-serif;
        color: #333333;
        line-height: 1.6;
    }
    .main {
        background-color: #f4f4f4;
        padding: 20px;
    }
    /* Responsive Layout */
    @media (max-width: 768px) {
        .home-header, .prediction-header, .about-header, .section-header {
            font-size: 24px;
        }
        .section-content, .footer {
            font-size: 14px;
        }
        .stButton button {
            font-size: 14px;
            padding: 8px 16px;
        }
    }
    /* Buttons and Inputs */
    .stButton button {
        background-color: #e41303;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #bf1202;
    }
    .stTextInput input {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #dddddd;
        width: 100%;
    }
    /* Headers */
    .home-header, .prediction-header, .about-header, .section-header {
        color: #e41303;
        text-align: center;
        font-size: 32px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .section-header {
        color: #1e7910;
        font-size: 24px;
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    /* Content */
    .section-content {
        color: #555555;
        font-size: 16px;
        margin-bottom: 15px;
    }
    .footer {
        color: #e41303;
        font-size: 14px;
        margin-top: 30px;
        text-align: right;
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .sidebar .stButton button {
        background-color: #1e7910;
        color: white;
        margin-bottom: 10px;
    }
    /* Custom Containers */
    .custom-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .custom-container h1, .custom-container h2 {
        color: #e41303;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the class names for the tomato diseases
class_names = [
    "bacterial spot", "early blight", "healthy tomato", "late blight",
    "southern blight"
]

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('tomato_disease_db.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB,
            predicted_class TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Insert data into the database
def insert_prediction(image_data, predicted_class, confidence):
    conn = sqlite3.connect('tomato_disease_db.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (image, predicted_class, confidence)
        VALUES (?, ?, ?)
    ''', (image_data, predicted_class, confidence))
    conn.commit()
    conn.close()

# Register a new user
def register_user(first_name, last_name, username, password):
    conn = sqlite3.connect('tomato_disease_db.db')
    c = conn.cursor()
    try:
        # Hash the password before storing it
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        c.execute('''
            INSERT INTO users (first_name, last_name, username, password)
            VALUES (?, ?, ?, ?)
        ''', (first_name, last_name, username, hashed_password))
        conn.commit()
        st.success("Registration successful. Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose a different username.")
    conn.close()

# Authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect('tomato_disease_db.db')
    c = conn.cursor()
    c.execute('''
        SELECT first_name, last_name, password FROM users WHERE username = ?
    ''', (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[2]):
        return user
    return None

# Login Page
def login_page():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.full_name = f"{user[0]} {user[1]}"
            st.success(f"Login successful. Welcome, {st.session_state.full_name}!")
        else:
            st.error("Invalid username or password")

# Registration Page
def registration_page():
    st.header("Register")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if first_name and last_name and username and password:
            register_user(first_name, last_name, username, password)
        else:
            st.error("Please fill in all fields")

@st.cache_data
def load_model():
    model_path = "Project_Improved_Model2.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file was not found at: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

# Define the tomato disease solution function
def tomato_disease_solution(disease):
    solutions = {
        "bacterial spot": "Bacterial Spot Solution:\n"
                          "1. Use certified disease-free seeds.\n"
                          "2. Avoid overhead watering to reduce leaf wetness.\n"
                          "3. Apply copper-based bactericides as a preventative measure.\n"
                          "4. Remove and destroy infected plant debris.\n"
                          "5. Maintain proper plant spacing for air circulation.\n",
        "early blight": "Early Blight Solution:\n"
                        "1. Rotate crops to prevent pathogen buildup in the soil.\n"
                        "2. Use resistant tomato varieties.\n"
                        "3. Remove and destroy affected plant parts.\n"
                        "4. Apply fungicides like chlorothalonil or copper-based sprays.\n"
                        "5. Ensure proper plant spacing for good air circulation.\n",
        "healthy tomato": "Healthy Tomato Maintenance:\n"
                          "1. Ensure proper watering - water at the base, not overhead.\n"
                          "2. Use mulch to retain soil moisture and prevent soil-borne diseases.\n"
                          "3. Fertilize regularly with balanced nutrients.\n"
                          "4. Prune to promote good air circulation.\n"
                          "5. Monitor plants regularly for any signs of disease or pests.\n",
        "late blight": "Late Blight Solution:\n"
                       "1. Use resistant tomato varieties.\n"
                       "2. Remove and destroy infected plants immediately.\n"
                       "3. Apply fungicides containing mancozeb or chlorothalonil.\n"
                       "4. Avoid overhead watering to minimize moisture.\n"
                       "5. Practice crop rotation and soil sanitation.\n",
        "southern blight": "Southern Blight Solution:\n"
                           "1. Rotate crops to avoid soilborne pathogens.\n"
                           "2. Apply organic mulch to suppress the disease.\n"
                           "3. Remove and destroy infected plants and debris.\n"
                           "4. Use fungicides such as PCNB or flutolanil if necessary.\n"
                           "5. Ensure good soil drainage to prevent excess moisture.\n"
    }
    return solutions.get(disease, "No solution found for the detected disease.")

# Function to preprocess the image and predict the disease
def predict(model, img):
    # Preprocess the image
    img = img.resize((256, 256))  # Resize to match model input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make predictions
    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = round(100 * np.max(predictions[0]), 2)
        disease_solution = tomato_disease_solution(predicted_class)
        return predicted_class, confidence, disease_solution
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None

# Prediction page content
def prediction_page():
    st.header("Capture/Upload a Tomato Leaf Image for Disease Prediction 🌿🔍")
    
    model = load_model()  # Load your trained model

    # Image capture/upload options with columns
    col1, col2 = st.columns([1, 1])

    with col1:
        camera_file = st.camera_input("📸 Take a Picture")

    with col2:
        uploaded_file = st.file_uploader("🔄 Choose an Image", type=["jpg", "jpeg", "png"])

    img = None

    if camera_file is not None:
        img = Image.open(io.BytesIO(camera_file.getvalue()))
    elif uploaded_file is not None:
        img = Image.open(uploaded_file)

    if img:
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Perform prediction
        predicted_class, confidence, disease_solution = predict(model, img)
                                                    
        if predicted_class and confidence:
            st.write(f"### Prediction: **{predicted_class.capitalize()}**")
            st.write(f"### Confidence: **{confidence:.2f}%**")
            st.write("### Solution:")
            st.write(disease_solution)
            
            # Save the image and prediction to the database
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_data = img_byte_arr.getvalue()
            insert_prediction(img_data, predicted_class, confidence)
    else:
        st.warning("Please upload an image or capture one using your camera.")


# About page content
def about_page():
    st.header("About Our Team")
    st.write("""
    Our team is composed of experts in data science, machine learning, plant pathology, and experienced farmers. We are dedicated to providing
    innovative solutions to help farmers identify and manage tomato plant diseases effectively.
    """)

# Home page content
def home_page():
    st.header("Welcome to the Tomato Plant Disease Classification System")
    st.write("""
    This system helps farmers and gardeners identify common tomato plant diseases through image classification.
    Simply upload a photo of a tomato leaf, and our model will predict the disease and provide solutions.
    """)

def main():
    init_db()

    st.sidebar.header("Tomato Plant Disease Classification")
    st.sidebar.write("Please select a page:")
    page = st.sidebar.selectbox("", ["Home", "Predict", "About", "Login", "Register"])
    
    if page == "Home":
        home_page()
    elif page == "Predict":
        if st.session_state.get("logged_in", False):
            prediction_page()
        else:
            st.warning("Please log in to access the prediction page.")
            login_page()
    elif page == "About":
        about_page()
    elif page == "Login":
        login_page()
    elif page == "Register":
        registration_page()

if __name__ == "__main__":
    main()
