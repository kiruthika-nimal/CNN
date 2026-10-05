import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("digit_recognition_model.pkl")
    hog_config = joblib.load("hog_config.pkl")
    return model, hog_config


model, hog_config = load_model()


# ============================================================
# HOG FEATURE EXTRACTION
# ============================================================

def extract_hog_features(image):

    features = hog(
        image,
        orientations=hog_config["orientations"],
        pixels_per_cell=hog_config["pixels_per_cell"],
        cells_per_block=hog_config["cells_per_block"],
        block_norm=hog_config["block_norm"]
    )

    return features.reshape(1, -1)


# ============================================================
# TITLE
# ============================================================

st.title("🔢 Handwritten Digit Recognition")

st.markdown(
    """
    Upload an image of a handwritten digit and let the
    machine learning model predict the digit.
    """
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Convert to grayscale
    image = image.convert("L")

    # Display original image
    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        caption="Uploaded handwritten digit",
        width=200
    )

    # Resize to 28 × 28
    image = image.resize((28, 28))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # ========================================================
    # EXTRACT HOG FEATURES
    # ========================================================

    hog_features = extract_hog_features(image_array)

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(hog_features)[0]

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction")

    st.success(
        f"Predicted Digit: {prediction}"
    )