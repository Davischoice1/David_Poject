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
    
def tomato_disease_solution(disease_name):
    solutions = {
        "Bacterial Spot": {
            "Immediate Actions": [
                "Remove Infected Leaves:\nCarefully prune and dispose of leaves showing signs of bacterial spot to reduce the spread of the bacteria.",
                "Improve Air Circulation:\nSpace plants properly and remove excess foliage to promote better airflow, which helps to reduce moisture on the leaves."
            ],
            "Long-term Solutions": [
                "Apply Copper-Based Fungicides:\nRegularly apply copper-based fungicides to protect healthy plants, especially during wet weather.",
                "Use Disease-Resistant Varieties:\nSelect tomato varieties known to be resistant to bacterial spot for future plantings.",
                "Sanitize Tools and Equipment:\nDisinfect garden tools after use to prevent the spread of bacteria to other plants."
            ]
        },
        "Early Blight": {
            "Immediate Actions": [
                "Remove and Destroy Infected Plant Parts:\nCut off and dispose of any leaves or stems showing symptoms of early blight to limit the spread.",
                "Apply Fungicides:\nBegin treatment with fungicides containing chlorothalonil or copper at the first sign of the disease. Reapply as recommended by the product label."
            ],
            "Long-term Solutions": [
                "Mulch Around Plants:\nApply mulch to reduce soil splash, which can spread the blight from soil to leaves.",
                "Rotate Crops:\nAvoid planting tomatoes or related crops in the same location for at least two to three years to reduce the presence of the pathogen in the soil.",
                "Plant Resistant Varieties:\nChoose tomato varieties that are resistant to early blight."
            ]
        },
        "Healthy Tomato": {
            "Immediate Actions": [
                "Maintain Regular Monitoring:\nContinuously inspect plants for any early signs of disease or pests. Early detection can prevent major outbreaks.",
                "Ensure Proper Watering:\nWater the plants at the base rather than overhead to keep the foliage dry and reduce the risk of disease."
            ],
            "Long-term Solutions": [
                "Practice Crop Rotation:\nRotate tomato crops with non-susceptible crops to minimize the buildup of soil-borne diseases.",
                "Use Disease-Resistant Varieties:\nSelect varieties that are naturally resistant to common tomato diseases.",
                "Fertilize Appropriately:\nProvide balanced nutrients to keep the plants healthy and more resilient to disease."
            ]
        },
        "Late Blight": {
            "Immediate Actions": [
                "Remove and Destroy Affected Plants:\nIf late blight is detected, remove and destroy infected plants immediately to prevent the disease from spreading.",
                "Apply Fungicides:\nUse fungicides with active ingredients like chlorothalonil, mancozeb, or copper, applying them according to the product label instructions."
            ],
            "Long-term Solutions": [
                "Monitor Weather Conditions:\nKeep an eye on weather forecasts, as late blight thrives in cool, wet conditions. Apply fungicides preventatively if these conditions are expected.",
                "Use Resistant Varieties:\nPlant tomato varieties that are resistant to late blight to reduce the risk of infection.",
                "Practice Crop Rotation:\nRotate your tomato crops to different areas each year to avoid building up the pathogen in the soil."
            ]
        },
        "Southern Blight": {
            "Immediate Actions": [
                "Remove Infected Plants:\nAs soon as southern blight is identified, remove and destroy infected plants to prevent the fungus from spreading.",
                "Apply Soil Fungicides:\nTreat the soil around healthy plants with fungicides like PCNB (pentachloronitrobenzene) to protect them from infection."
            ],
            "Long-term Solutions": [
                "Soil Solarization:\nSolarize the soil during the off-season by covering it with clear plastic for 4-6 weeks. This helps to kill the fungus in the upper layers of soil.",
                "Rotate Crops:\nRotate with non-susceptible crops (e.g., corn, grains) to reduce the buildup of the pathogen in the soil.",
                "Apply Organic Mulch:\nUse organic mulches around the base of plants to create a barrier between the soil and the plant stems."
            ]
        }
    }

    return solutions.get(disease_name, "No solution available for the specified disease.")


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
  # About Page Header
        st.markdown("<div class='about-page'><h1 class='about-header'>About the Tomato Plant Disease Classification System</h1></div>", unsafe_allow_html=True)

        # Mission Statement
        st.markdown("<div class='section-header'>Our Mission</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='section-content'>
            Our mission is to empower farmers, gardeners, and researchers with advanced technology to identify and manage diseases affecting tomato plants swiftly and accurately. By leveraging state-of-the-art machine learning algorithms, we aim to enhance plant health and increase crop yields, ensuring a sustainable future for agriculture.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Goal
        st.markdown("<div class='section-header'>Our Goal</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='section-content'>
            The primary goal of our Tomato Plant Disease Classification System is to provide a reliable, user-friendly tool that can diagnose tomato plant diseases from images. We strive to offer actionable insights and effective solutions, helping users to take timely actions to protect their crops and improve overall plant health.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Dataset
        st.markdown("<div class='section-header'>The Dataset</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='section-content'>
            Our system utilizes a comprehensive dataset of tomato plant images that include various disease categories. The dataset is curated from reliable sources and includes a diverse range of images to ensure accurate and robust disease detection. We continuously update and expand the dataset to improve the system's performance and adapt to new disease strains.
            </div>
            """,
            unsafe_allow_html=True
        )

        # The Team
        st.markdown("<div class='section-header'>Meet The Team</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='section-content'>
            <div class='Expert Team'>

            Our team behind the Tomato Plant Disease Classification System consists of experts in data science, machine learning, plant pathology, and experienced farmers. Together, we work to revolutionize the detection and management of tomato plant diseases. The data scientists and machine learning engineers develop and refine algorithms to accurately identify diseases from images, while the plant pathologist ensures scientific accuracy. Farmers provide practical insights, shaping actionable solutions. The team's collaborative approach blends technology with agricultural expertise, continuously innovating to offer a user-friendly tool for farmers and gardeners. We are dedicated to supporting healthier, more productive tomato crops and are open to collaboration and inquiries.

            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Contact Us
        st.markdown("<div class='section-header'>Contact Us</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='contact-info'>
            <p>For any inquiries or support, please reach out to us:</p>
            <ul>
                <li><strong>Email:</strong> <a href="mailto:support@tomatodiseaseclassifier.com">support@tomatodiseaseclassifier.com</a></li>
                <li><strong>Phone:</strong> +234-7064206404</li>
                <li><strong>Address:</strong> PMB 704, Ondo State, Nigeria</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Additional Information
        st.markdown("<div class='section-header'>Additional Information</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='section-content'>
            <p>We are constantly working to enhance our system's capabilities and to provide more value to our users. Follow us on our social media channels for updates and tips on plant health. \ntwitter: @DAVISCHOICE4 \nFacebook: @david.omeiza.92</p>
            </div>
            """,
            unsafe_allow_html=True
        )
# Home page content
def home_page():
    logo = Image.open("toma.jpg")
    st.image(logo, use_column_width=True)
    st.header("Welcome to the Tomato Plant Disease Classification System 🌿🔍")
    st.write("""
    This system helps farmers and gardeners identify common tomato plant diseases through image classification.
    Simply upload a photo of a tomato leaf, and our model will predict the disease and provide solutions.
    """)

def main():
    init_db()
    
    st.sidebar.image("Logo.jpg", width=100)
    st.sidebar.header("Dashboard")
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
