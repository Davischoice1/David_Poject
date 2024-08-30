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

# Login Page
def login_page():
    st.markdown('<h1 class="header">Login</h1>', unsafe_allow_html=True)
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
    st.markdown('<h1 class="header">Register</h1>', unsafe_allow_html=True)
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if first_name and last_name and username and password:
            register_user(first_name, last_name, username, password)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.full_name = f"{first_name} {last_name}"
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
        "bacterial spot": "**Bacterial Spot Solution:**\n"
                          "\n**Immediate Actions:**\n"
                          "- Remove Infected Leaves: Carefully prune and dispose of leaves showing signs of bacterial spot to reduce the spread of the bacteria.\n"
                          "- Improve Air Circulation: Space plants properly and remove excess foliage to promote better airflow, which helps to reduce moisture on the leaves.\n\n"
                          "**Long-term Solutions:**\n"
                          "- Apply Copper-Based Fungicides: Regularly apply copper-based fungicides to protect healthy plants, especially during wet weather.\n"
                          "- Use Disease-Resistant Varieties: Select tomato varieties known to be resistant to bacterial spot for future plantings.\n"
                          "- Sanitize Tools and Equipment: Disinfect garden tools after use to prevent the spread of bacteria to other plants.\n",

        "early blight": "**Early Blight Solution:**\n"
                        "\n**Immediate Actions:**\n"
                        "- Remove and Destroy Infected Plant Parts: Cut off and dispose of any leaves or stems showing symptoms of early blight to limit the spread.\n"
                        "- Apply Fungicides: Begin treatment with fungicides containing chlorothalonil or copper at the first sign of the disease. Reapply as recommended by the product label.\n\n"
                        "**Long-term Solutions:**\n"
                        "- Mulch Around Plants: Apply mulch to reduce soil splash, which can spread the blight from soil to leaves.\n"
                        "- Rotate Crops: Avoid planting tomatoes or related crops in the same location for at least two to three years to reduce the presence of the pathogen in the soil.\n"
                        "- Plant Resistant Varieties: Choose tomato varieties that are resistant to early blight.\n",

        "healthy tomato": "**Healthy Tomato Maintenance:**\n"
                          "\n**Immediate Actions:**\n"
                          "- Maintain Regular Monitoring: Continuously inspect plants for any early signs of disease or pests. Early detection can prevent major outbreaks.\n"
                          "- Ensure Proper Watering: Water the plants at the base rather than overhead to keep the foliage dry and reduce the risk of disease.\n\n"
                          "**Long-term Solutions:**\n"
                          "- Practice Crop Rotation: Rotate tomato crops with non-susceptible crops to minimize the buildup of soil-borne diseases.\n"
                          "- Use Disease-Resistant Varieties: Select varieties that are naturally resistant to common tomato diseases.\n"
                          "- Fertilize Appropriately: Provide balanced nutrients to keep the plants healthy and more resilient to disease.\n",

        "late blight": "**Late Blight Solution:**\n"
                       "\n**Immediate Actions:**\n"
                       "- Remove and Destroy Affected Plants: If late blight is detected, remove and destroy infected plants immediately to prevent the disease from spreading.\n"
                       "- Apply Fungicides: Use fungicides with active ingredients like chlorothalonil, mancozeb, or copper, applying them according to the product label instructions.\n\n"
                       "**Long-term Solutions:**\n"
                       "- Monitor Weather Conditions: Keep an eye on weather forecasts, as late blight thrives in cool, wet conditions. Apply fungicides preventatively if these conditions are expected.\n"
                       "- Use Resistant Varieties: Plant tomato varieties that are resistant to late blight to reduce the risk of infection.\n"
                       "- Practice Crop Rotation: Rotate your tomato crops to different areas each year to avoid building up the pathogen in the soil.\n",

        "southern blight": "**Southern Blight Solution:**\n"
                           "\n**Immediate Actions:**\n"
                           "- Remove Infected Plants: As soon as southern blight is identified, remove and destroy infected plants to prevent the fungus from spreading.\n"
                           "- Apply Soil Fungicides: Treat the soil around healthy plants with fungicides like PCNB (pentachloronitrobenzene) to protect them from infection.\n\n"
                           "**Long-term Solutions:**\n"
                           "- Soil Solarization: Solarize the soil during the off-season by covering it with clear plastic for 4-6 weeks. This helps to kill the fungus in the upper layers of soil.\n"
                           "- Rotate Crops: Rotate with non-susceptible crops (e.g., corn, grains) to reduce the buildup of the pathogen in the soil.\n"
                           "- Apply Organic Mulch: Use organic mulches around the base of plants to create a barrier between the soil and the plant stems.\n"
    }
    return solutions.get(disease.lower(), "No solution available for the specified disease.")

# Function to preprocess the image and predict the disease
def predict(model, img):
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

# Custom CSS for styling
st.markdown("""
    <style>
        .header {
            font-size: 36px;
            font-weight: bold;
            color: #006400;
        }
        .container {
            background-color: #F0F8F0;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        .btn-primary {
            background-color: #006400;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }
        .btn-primary:hover {
            background-color: #004d00;
        }
        .sidebar .sidebar-content {
            background-color: #003d00;
            color: white;
        }
        .sidebar .sidebar-content a {
            color: #8FBC8F;
        }
        .sidebar .sidebar-content a:hover {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("logo.png", width=150)
    st.title("Tomato Disease Classifier")
    option = st.selectbox("Choose an option:", ["Home", "About", "FAQ", "Login", "Register"])

# Page Routing
if option == "Home":
    st.markdown('<h1 class="header">Welcome to the Tomato Disease Classifier 🌿🔍</h1>', unsafe_allow_html=True)
    st.write("""
    **Capture or Upload an Image of a Tomato Leaf** to diagnose the disease and receive actionable solutions.
    """)

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

        
    if uploaded_file:
        model = load_model()
        image = Image.open(uploaded_file)
        predicted_class, confidence = predict(model, image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f"**Predicted Disease:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(tomato_disease_solution(predicted_class))
        # Save prediction to the database
        image_data = uploaded_file.read()
        insert_prediction(image_data, predicted_class, confidence)

elif option == "About":
    st.markdown('<h1 class="header">About Us</h1>', unsafe_allow_html=True)
    st.write("Our team consists of data scientists, machine learning experts, pathologists, and experienced farmers.")
    st.write("We aim to provide an efficient solution for tomato plant disease detection to support farmers.")

elif option == "FAQ":
    st.markdown('<h1 class="header">FAQ</h1>', unsafe_allow_html=True)
    st.write("**Q: What types of tomato diseases can you detect?**")
    st.write("A: We can detect bacterial spot, early blight, late blight, and southern blight.")
    st.write("**Q: How accurate is the prediction?**")
    st.write("A: Our model provides predictions with high accuracy. Confidence scores are displayed with each prediction.")
    st.write("**Q: How should I handle the diseases detected?**")
    st.write("Refer to the disease solutions provided for guidance on managing and treating detected diseases.")

elif option == "Login":
    login_page()

elif option == "Register":
    registration_page()

# Footer
st.markdown("""
    <footer style="text-align: center; padding: 20px; background-color: #003d00; color: white;">
        <p>&copy; 2024 Tomato Disease Classifier. All rights reserved.</p>
    </footer>
""", unsafe_allow_html=True)
