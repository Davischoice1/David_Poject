# Importing the dependencies
import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import os
import bcrypt
from PIL import Image
import io

# Custom CSS styling
css = """
<style>
body {
    background-color: #8acc76; /* Leaf green background */
    color: #000000; /* White text color */
    font-family: Arial, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    color: #000000; /* Light goldenrod yellow */
}

.stButton>button {
    background-color: #556B2F; /* Dark olive green for buttons */
    color: #FFFFFF;
    padding: 10px;
    border-radius: 5px;
    border: none;
    }

.stSidebar {
    background-color: #286c13 ; /* Dark sea green for sidebar */
}

.stSidebar .stButton>button {
    background-color: #556B2F; /* Dark olive green for sidebar buttons */
}

.stSidebar .stButton>button:hover {
    background-color: #6B8E23; /* Olive drab for sidebar buttons on hover */
}

.container {
    padding: 20px;
    background-color: rgba(255, 255, 255, 0.1); /* Transparent white for container background */
    border-radius: 10px;
    margin-bottom: 20px;
}

.about-header, .section-header {
    font-weight: bold;
    font-size: 24px;
    color: #000000;
    margin-bottom: 10px;
}

.section-content {
    font-size: 16px;
    color: #000000;
    line-height: 1.6;
    margin-bottom: 20px;
}

.faq-header {
    font-size: 28px;
    font-weight: bold;
    color: #000000;
    margin-bottom: 20px;
}

.faq-item {
    margin-bottom: 20px;
}

.faq-question {
    font-size: 18px;
    font-weight: bold;
    color: #000000;
    margin-bottom: 5px;
}

.faq-answer {
    font-size: 16px;
    color: #000000;
    line-height: 1.6;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

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
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "About", 'FAQ', "Logout"])

    if app_mode == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.full_name = None
        st.success("You have been logged out.")

    if app_mode == "Home":
    # Display logo with styling
      logo = Image.open("toma.jpg")
      st.image(logo, width=200, caption="Tomato Plant Disease Classification System", use_column_width=True)

      # Main header
      st.header("Welcome to the Tomato Plant Disease Classification System! 🍅🔍")

      # Introduction text
      st.write("""
      **Detect and Manage Tomato Plant Diseases** with ease using our cutting-edge machine learning technology. Our system helps you diagnose diseases from images of your tomato plants and provides actionable solutions to keep your plants healthy and thriving.
      """)

      # Get Started section with call-to-action
      st.markdown("### 🚀 Get Started")
      st.write("""
      1. **Upload an Image:** Navigate to the **Disease Recognition** page to upload an image of your tomato plant.
      2. **Receive Diagnosis:** Our system will analyze the image and provide a diagnosis along with actionable solutions.
      3. **Explore More:** Visit the **Home** page to learn more about our system and its capabilities.
      """)

      # Footer or additional information
      st.markdown("""
      **Need Assistance?** Navigate through the Dashboard and Visit our **[FAQ]** or **[About Us]** for support.
      """)


    elif app_mode == "Prediction":
      st.subheader("Tomato Leaf Disease Detection 🌿🔍")
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

          # Perform prediction
          with st.spinner("Classifying..."):
              predicted_class, confidence, disease_solution = predict(model, img)
              
              if predicted_class:
                  success_message = "Prediction: " + predicted_class + " (" + str(confidence) + "% confidence)"
                  st.success(success_message)
                  st.info(disease_solution)

              #if predicted_class:
                  #st.success(predicted_class, confidence% )
                  #st.info(disease_solution)
      else:
          st.warning("Please upload an image or capture one using your camera.")

    elif app_mode == "About":
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
          <p>We are constantly working to enhance our system's capabilities and to provide more value to our users. Follow us on our social media channels for updates and tips on plant health.</p>
          <ul>
              <li><strong>Twitter:</strong> <a href="https://twitter.com/DAVISCHOICE4" target="_blank">@DAVISCHOICE4</a></li>
              <li><strong>Facebook:</strong> <a href="https://www.facebook.com/david.omeiza.92" target="_blank">@david.omeiza.92</a></li>
          </ul>
          </div>
          """,
          unsafe_allow_html=True
      )


    elif app_mode == "FAQ":
      # FAQ Header
      st.markdown("<h2 class='faq-header'>Frequently Asked Questions (FAQ)</h2>", unsafe_allow_html=True)

      # FAQ Items
      faqs = [
          {
              "question": "1. How do I use the Tomato Disease Classification System?",
              "answer": "To use the system, simply upload an image of your tomato plant leaf or take a picture using the built-in camera feature. The system will analyze the image and provide you with a diagnosis of the disease and suggested solutions."
          },
          {
              "question": "2. What types of tomato diseases can the system detect?",
              "answer": "The system can detect a variety of tomato diseases, including bacterial spot, early blight, late blight, and southern blight. It also provides solutions to manage these diseases effectively."
          },
          {
              "question": "3. Is there a way to save my prediction results?",
              "answer": "Yes, all your predictions are automatically saved in the database. You can view your past predictions and their details in the database if you are logged in."
          },
          {
              "question": "4. How do I register and log in to the system?",
              "answer": "To register, go to the Registration page and fill in your details. After registration, you can log in using your username and password. If you already have an account, use the Login page to access the system."
          },
          {
              "question": "5. What should I do if I encounter any issues or need support?",
              "answer": "For any issues or support, you can contact us via email at support@tomatodiseaseclassifier.com or call us at +234-7064206404. We are here to help you with any questions or concerns."
          }
      ]

      # Display FAQ Items
      for faq in faqs:
          with st.container():
              st.markdown(f"<div class='faq-item'><p class='faq-question'>{faq['question']}</p><p class='faq-answer'>{faq['answer']}</p></div>", unsafe_allow_html=True)

else:
    login_mode = st.sidebar.selectbox("Login or Register", ["Login", "Register"])

    if login_mode == "Login":
        login_page()
    elif login_mode == "Register":
        registration_page()

