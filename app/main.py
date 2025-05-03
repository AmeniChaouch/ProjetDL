import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from keras.models import load_model

# === Initialisation du modèle ===
working_dir = os.path.dirname(os.path.abspath(__file__))
#model_path = os.path.join(working_dir, 'trained_model', 'trained_fashion_mnist_model.h5')
#model_path = os.path.join(working_dir, 'trained_model', 'trained_fashion_mnist_augmented.h5')

model_path = os.path.join(working_dir, 'trained_model', 'trained_fashion_mnist_transfer.h5')

if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
else:
    st.error(f"Fichier introuvable : {model_path}")
    st.stop()

# === Labels ===
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# === Prétraitement image ===
#def preprocess_image(image):
    #img = image.resize((28, 28)).convert('L')  # grayscale
    #img_array = np.array(img) / 255.0
    #img_array = img_array.reshape((1, 28, 28, 1))
    #return img_array
def preprocess_image(image):  # image est déjà ouverte avec PIL.Image.open()
    img = image.resize((96, 96))       # Redimensionner en 96x96
    img = img.convert('RGB')           # Convertir en RGB
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 96, 96, 3))  # Format attendu par MobileNet
    return img_array


# === App Streamlit ===
st.title(' Fashion Item Classifier')

option = st.radio("", [ "Tester une image"])

# === OPTION 1 : UPLOAD ===
if option == "":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image.resize((100, 100)), caption="Image uploadée")

        if st.button("Classify l'image uploadée"):
            img_array = preprocess_image(image)
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            st.success(f"Prediction: {class_names[predicted_class]}")

# === OPTION 2 : IMAGE LOCALE ===
else:
    test_images_dir = os.path.join("C:/Users/ASUS/Downloads/fashion-mnist-end-to-end-project-main (1)/fashion-mnist-end-to-end-project-main", "test_images")
    if os.path.exists(test_images_dir):
        image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        selected_file = st.selectbox("Sélectionne une image :", image_files)

        if selected_file:
            image_path = os.path.join(test_images_dir, selected_file)
            image = Image.open(image_path)
            st.image(image.resize((100, 100)), caption=selected_file)

            if st.button("Classify l'image locale"):
                img_array = preprocess_image(image)
                result = model.predict(img_array)
                predicted_class = np.argmax(result)
                st.success(f"Prediction: {class_names[predicted_class]}")
    else:
        st.error("Le dossier 'test_images' n'existe pas.")
