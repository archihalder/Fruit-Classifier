import time
import urllib
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


# @st.cache(allow_output_mutation=True, suppress_st_warning=True)

html_temp = """
    <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Fruit Classifier</h1></center>
    
    </div>
    """

st.markdown(html_temp, unsafe_allow_html=True)
html_temp = """
    <div>
    <h2></h2>
    <center><h3>Please upload any Fruit Image from the given list</h3></center>
    <center><h3> [Apple, Banana, Orange, Mango, Pineapple] </h3></center>
    </div>
    """

st.set_option("deprecation.showfileUploaderEncoding", False)
st.markdown(html_temp, unsafe_allow_html=True)

opt = st.selectbox(
    "How do you want to upload the image for classification?\n",
    ("Please Select", "Upload image via link", "Upload image from device"),
)
if opt == "Upload image from device":
    file = st.file_uploader("Select", type=["jpg", "png", "jpeg"])
    st.set_option("deprecation.showfileUploaderEncoding", False)
    if file is not None:
        image = Image.open(file)

elif opt == "Upload image via link":

    try:
        img = st.text_input("Enter the Image Address")
        image = Image.open(urllib.request.urlopen(img))

    except:
        if st.button("Submit"):
            show = st.error("Please Enter a valid Image Address!")
            time.sleep(4)
            show.empty()

if image is not None:

    try:
        st.image(image, width=300, caption="Uploaded Image")

        if st.button("Classify"):
            img_array = np.array(image.resize((128, 128), Image.ANTIALIAS))
            img_array = np.array(img_array, dtype="uint8")
            img_array = np.array(img_array) / 255.0

            model_dir = "Model/model.h5"
            model = keras.models.load_model(model_dir)

            # Labels
            train_labels = {
                "Apple": 0,
                "Banana": 1,
                "Mango": 2,
                "Orange": 3,
                "Pineapple": 4,
            }
            labels = dict((value, key) for key, value in train_labels.items())

            # Predicting
            predictions = model.predict(img_array[np.newaxis, ...])
            acc = np.max(predictions[0]) * 100
            result = labels[np.argmax(predictions[0], axis=-1)]

            # Displaying output
            st.info(
                f'The uploaded image has been classified as " {result}" with confidence {acc}%.'
            )

    except:
        st.success("Please enter an Input Image of an appropriate format :) ")