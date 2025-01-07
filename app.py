import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="Potato Disease Classification")

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Use updated caching function
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('potato_model.h5', compile=False)  # Avoid compilation issues
    return model

with st.spinner('Model is being loaded...'):
    model = load_model()

st.write("""
# Potato Disease Classification
""")

# File uploader for user to upload an image
file = st.file_uploader("Upload a potato leaf image (jpg or png)", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (256, 256)  # Resize to match the model's input shape
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Use LANCZOS for resampling
    img = np.asarray(image) / 255.0  # Normalize pixel values
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    predictions = import_and_predict(image, model)
    class_names = ['Early blight', 'Late blight', 'Healthy']
    prediction_class = class_names[np.argmax(predictions)]
    string = f"Prediction: {prediction_class}"

    if prediction_class == 'Healthy':
        st.success(string)
    else:
        st.warning(string)
