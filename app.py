import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load the model 
MODEL_PATH = 'fruit_classifier_resnet50.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names as used during training
class_names = [
    "F_Banana", "F_Lemon", "F_Lulo", "F_Mango", "F_Orange", "F_Strawberry", "F_Tamarillo", "F_Tomato",
    "S_Banana", "S_Lemon", "S_Lulo", "S_Mango", "S_Orange", "S_Strawberry", "S_Tamarillo", "S_Tomato"
]

supported_fruits = [
    "Banana",
    "Lemon",
    "Lulo",
    "Mango",
    "Orange",
    "Strawberry",
    "Tamarillo",
    "Tomato"
]

# Streamlit app title and description
st.set_page_config(page_title="Fruit Freshness Classifier", page_icon="🍎", layout="wide")
st.title("🍎 Fruit Freshness Classifier")
st.markdown(
    "<h3 style='color:#54B4D3;'>Upload a fruit image, and find out if it's fresh or spoiled!</h3>",
    unsafe_allow_html=True
)

st.markdown(
    f"<b style='color:#44a;'>Only the following fruits are supported:</b> " +
    ", ".join(f"<span style='color:#4bb543'><b>{f}</b></span>" for f in supported_fruits),
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', width=300, use_container_width=False)  # Show smaller image
    st.markdown("---")
    st.write("Classifying...")

    # Preprocess and predict
    x = preprocess_image(img)
    predictions = model.predict(x)
    pred_idx = np.argmax(predictions, axis=1)[0]
    
    result = class_names[pred_idx]
    confidence = float(np.max(tf.nn.softmax(predictions)))

    # Separate quality and fruit name
    quality_map = {"F": "Fresh", "S": "Spoiled"}
    if "_" in result:
        quality_code, fruit = result.split("_", 1)
        quality = quality_map.get(quality_code, "Unknown")
        display_result = f"{quality} {fruit}"
    else:
        display_result = result

    st.write(f"### This looks like: **{display_result}**")

    if result.startswith("F_"):
        st.success("The fruit is likely FRESH!")
    elif result.startswith("S_"):
        st.error("The fruit looks SPOILED!")
    else:
        st.info("Can't determine quality.")

else:
    st.info("🖼️ Please upload a fruit image.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Made with ❤️ using Streamlit & TensorFlow")
