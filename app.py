import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


@st.cache(allow_output_mutation=True)
def loading_model():
    # Loading the model
    model_dir = "/home/archihalder/Projects/Fruit Classifier/Model/model.h5"
    loaded_model = keras.models.load_model(model_dir)
    return loaded_model


# Labels
train_labels = {"Apple": 0, "Banana": 1, "Mango": 2, "Orange": 3, "Pineapple": 4}


def magic(test_img, model):
    img = image.load_img(test_img, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.array(img_array) / 255.0

    # Getting the labels
    labels = dict((value, key) for key, value in train_labels.items())
    predictions = model.predict(img_array[np.newaxis, ...])

    # Displaying the output
    acc = np.max(predictions[0]) * 100
    print(f"Accuracy: {acc}%")
    result = labels[np.argmax(predictions[0], axis=-1)]
    print(f"Output: {result}")


def main():
    html_code = """
    <div>
        <h1 style="text-align: center"> Fruit Classifier </h1>
    </div>
    """

    st.markdown(html_code, unsafe_allow_html=True)

    # Upload the file
    uploaded_file = st.file_uploader("Choose an image", type=["jpeg", "png", "jpg"])
    st.set_option("deprecation.showfileUploaderEncoding", False)

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        img = Image.open(BytesIO(bytes_data))
        # img =
        st.image(img, caption="File Uploaded", use_column_width=True)

    model = loading_model()
    if st.button("Predict"):
        st.write("Predicting...")
        st.write(magic(img, model))


if __name__ == "__main__":
    main()