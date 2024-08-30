# Importing the dependencies
import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import os
import bcrypt
from PIL import Image
import io

# Define the class names for the tomato diseases
class_names = [
    "bacterial spot", "early blight", "healthy tomato", "late blight", "southern blight"
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

# Load the model
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
        "bacterial spot": """
            **Bacterial Spot Solution:**\n
            **Immediate Actions:**\n
            - Remove Infected Leaves: Carefully prune and dispose of leaves showing signs of bacterial spot to reduce the spread of the bacteria.\n
            - Improve Air Circulation: Space plants properly and remove excess foliage to promote better airflow, which helps to reduce moisture on the leaves.\n\n
            **Long-term Solutions:**\n
            - Apply Copper-Based Fungicides: Regularly apply copper-based fungicides to protect healthy plants, especially during wet weather.\n
            - Use Disease-Resistant Varieties: Select tomato varieties known to be resistant to bacterial spot for future plantings.\n
            - Sanitize Tools and Equipment: Disinfect garden tools after use to prevent the spread of bacteria to other plants.\n
        """,
        "early blight": """
            **Early Blight Solution:**\n
            **Immediate Actions:**\n
            - Remove and Destroy Infected Plant Parts: Cut off and dispose of any leaves or stems showing symptoms of early blight to limit the spread.\n
            - Apply Fungicides: Begin treatment with fungicides containing chlorothalonil or copper at the first sign of the disease. Reapply as recommended by the product label.\n\n
            **Long-term Solutions:**\n
            - Mulch Around Plants: Apply mulch to reduce soil splash, which can spread the blight from soil to leaves.\n
            - Rotate Crops: Avoid planting tomatoes or related crops in the same location for at least two to three years to reduce the presence of the pathogen in the soil.\n
            - Plant Resistant Varieties: Choose tomato varieties that are resistant to early blight.\n
        """,
        "healthy tomato": """
            **Healthy Tomato Maintenance:**\n
            **Immediate Actions:**\n
            - Maintain Regular Monitoring: Continuously inspect plants for any early signs of disease or pests. Early detection can prevent major outbreaks.\n
            - Ensure Proper Watering: Water the plants at the base rather than overhead to keep the foliage dry and reduce the risk of disease.\n\n
            **Long-term Solutions:**\n
            - Practice Crop Rotation: Rotate tomato crops with non-susceptible crops to minimize the buildup of soil-borne diseases.\n
            - Use Disease-Resistant Varieties: Select varieties that are naturally resistant to common tomato diseases.\n
            - Fertilize Appropriately: Provide balanced nutrients to keep the plants healthy and more resilient to disease.\n
        """,
        "late blight": """
            **Late Blight Solution:**\n
            **Immediate Actions:**\n
            - Remove and Destroy Affected Plants: If late blight is detected, remove and destroy infected plants immediately to prevent the disease from spreading.\n
            - Apply Fungicides: Use fungicides with active ingredients like chlorothalonil, mancozeb, or copper, applying them according to the product label instructions.\n\n
            **Long-term Solutions:**\n
            - Monitor Weather Conditions: Keep an eye on weather forecasts, as late blight thrives in cool, wet conditions. Apply fungicides preventatively if these conditions are expected.\n
            - Use Resistant Varieties: Plant tomato varieties that are resistant to late blight to reduce the risk of infection.\n
            - Practice Crop Rotation: Rotate your tomato crops to different areas each year to avoid building up the pathogen in the soil.\n
        """,
        "southern blight": """
            **Southern Blight Solution:**\n
            **Immediate Actions:**\n
            - Remove Infected Plants: As soon as southern blight is identified, remove and destroy infected plants to prevent the fungus from spreading.\n
            - Apply Soil Fungicides: Treat the soil around healthy plants with fungicides like PCNB (pentachloronitrobenzene) to protect them from infection.\n\n
            **Long-term Solutions:**\n
            - Soil Solarization: Solarize the soil during the off-season by covering it with clear plastic for 4-6 weeks. This helps to kill the fungus in the upper layers of soil.\n
            - Rotate Crops: Rotate with non-susceptible crops (e.g., corn, grains) to reduce the buildup of the pathogen in the soil.\n
            - Apply Organic Mulch: Use organic mulches around the base of plants to create a barrier between the soil and the plant stems.\n
        """
    }
    return solutions.get(disease.lower(), "No solution available for the specified disease.")

# Function to preprocess the image and predict the disease
def predict(model, img):
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

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

# Initialize the database
init_db()

# Load the model once and reuse it
model = load_model()

# Check login state and display appropriate content
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    st.sidebar.image("Logo.jpg", width=100)
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "About", "FAQ", "Logout"])

    if app_mode == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.full_name = None
        st.success("You have been logged out.")

        st.markdown("""
        <style>
        /* Main content background */
        .main {
            background-color: #0b1a02; /* Dark green background for the app */
        }
        
        /* Header styling for the main content */
        .header {
            font-size: 2.5em;
            color: #FFFFFF; /* White text color for headers */
            text-align: center;
            margin-top: 20px;
            font-weight: bold;
        }
    
        /* Container styling */
        .main-container {
            background-color: white;
        }
    
        /* Footer styling */
        .footer {
            text-align: center;
            font-size: 1em;
            color: white;
            margin-top: 20px;
        }
    
        /* Camera input styling */
        .stCameraInput > div {
            background-color: #1c4012; /* Dark green background */
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-weight: bold;
        }
    
        /* File uploader styling */
        .stFileUploader > div {
            background-color: #2e5a2f; /* Dark green background */
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-weight: bold;
        }
    
        /* Alert message styling */
        .stAlert {
            font-size: 1.1em; /* Slightly larger font size */
            color: white; /* White text color */
            border-radius: 10px;
            padding: 10px;
        }
    
        .stAlert.error {
            background-color: #f44336; /* Red background for error messages */
        }
    
        .stAlert.success {
            background-color: #4CAF50; /* Green background for success messages */
        }
        
        /* Sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #e6f4e8; /* Light green background color */
        }
    
        /* Sidebar title styling */
        [data-testid="stSidebar"] h1 {
            color: darkolivegreen; /* Dark olive green title color */
        }
    
        /* Sidebar selectbox styling */
        [data-testid="stSidebar"] .stSelectbox label {
            color: darkgreen; /* Dark green label color */
        }
    
        /* Sidebar selectbox options styling */
        [data-testid="stSidebar"] .stSelectbox div[data-testid="stMarkdownContainer"] {
            color: darkgreen; /* Dark green options color */
        }
        </style>
    """, unsafe_allow_html=True)  

    if app_mode == "Home":
        st.markdown('<div class="header">Tomato Plant Disease Classification System</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="main-container">
                <h2>Welcome to the Tomato Plant Disease Classification System</h2>
                <p>This system uses advanced machine learning algorithms to identify and classify tomato plant diseases. You can upload an image of your tomato leaf to receive a diagnosis and actionable solutions.</p>
            </div>
        """, unsafe_allow_html=True)

    elif app_mode == "Prediction":
        st.markdown('<div class="header">Disease Prediction</div>', unsafe_allow_html=True)

        # Image capture/upload options
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

            predicted_class, confidence, disease_solution = predict(model, img)
           
            # Display predicted class, confidence, and solution
            if predicted_class:
                st.success(f"**Predicted Disease:** {predicted_class}")
                st.success(f"**Confidence:** {confidence}%")
                st.info(f"**Solution:** {disease_solution}")

            
           # if predicted_class:
            #    st.success(f"**Predicted Disease:** {predicted_class}")
             #   st.success(f"**Confidence:** {confidence}%")
              #  st.info(f"**Solution:** {disease_solution}")
                
                # Insert prediction into the database
                image_data = uploaded_file.read()
                insert_prediction(image_data, predicted_class, confidence)
        else:
            st.write("Please upload an image to get the prediction.")

    elif app_mode == "About":
        st.markdown('<div class="header">About Us</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="main-container">
                <h2>About the Project</h2>
                <p>This project aims to provide accurate and timely diagnoses for tomato plant diseases. Our team includes data scientists, machine learning experts, pathologists, and experienced farmers working together to enhance agricultural practices.</p>
            </div>
        """, unsafe_allow_html=True)

    elif app_mode == "FAQ":
        st.markdown('<div class="header">Frequently Asked Questions</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="main-container">
                <h2>FAQ</h2>
                <p><strong>Q: What types of images can I upload?</strong></p>
                <p>A: You should upload images of tomato leaves. Images of other plants may not be accurately classified.</p>
                <p><strong>Q: How can I improve prediction accuracy?</strong></p>
                <p>A: Ensure the uploaded image is clear, with good lighting and focused on the tomato leaf.</p>
                <p><strong>Q: Can I trust the results?</strong></p>
                <p>A: While we strive for high accuracy, always consider additional diagnostic methods or consult with experts if needed.</p>
            </div>
        """, unsafe_allow_html=True)

else:
    st.sidebar.image("Logo.jpg", width=100)
    st.sidebar.title("Authentication")
    app_mode = st.sidebar.selectbox("Select Page", ["Login", "Register"])

    if app_mode == "Register":
        st.title("Register")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            if first_name and last_name and username and password:
                register_user(first_name, last_name, username, password)
            else:
                st.error("Please fill out all fields.")

    elif app_mode == "Login":
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.full_name = f"{user[0]} {user[1]}"
                st.success("Login successful. Welcome!")
            else:
                st.error("Invalid username or password.")

