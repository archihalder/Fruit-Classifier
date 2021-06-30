import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
import time
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


#@st.cache(allow_output_mutation=True, suppress_st_warning=True)

html_temp = '''
    <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Fruit Classifier</h1></center>
    
    </div>
    '''

st.markdown(html_temp, unsafe_allow_html=True)
html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Please upload Waste Image to find its Category</h3></center>
    </div>
    '''

st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(html_temp, unsafe_allow_html=True)

opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':

  try:
    img = st.text_input('Enter the Image Address')
    image = Image.open(urllib.request.urlopen(img))
    
  except:
    if st.button('Submit'):
      show = st.error("Please Enter a valid Image Address!")
      time.sleep(4)
      show.empty()

if image is not None:
    
    try:
        st.image(image, width = 300, caption = 'Uploaded Image')

        if st.button('Classify'):
            img_array = np.array(image.resize((128, 128), Image.ANTIALIAS))
            img_array = np.array(img_array, dtype='uint8')
            img_array = np.array(img_array) / 255.0

            model_dir = "Model/model.h5"
            model = keras.models.load_model(model_dir)
            
            # Labels
            train_labels = {"Apple": 0, "Banana": 1, "Mango": 2, "Orange": 3, "Pineapple": 4}

            # Getting the labels
            labels = dict((value, key) for key, value in train_labels.items())
            predictions = model.predict(img_array[np.newaxis, ...])
        
            acc = np.max(predictions[0]) * 100
            result = labels[np.argmax(predictions[0], axis=-1)]
            
            st.info('The uploaded image has been classified as " {}." with confidence {}%.'.format(result, acc)) 

    except:
        st.success("Please enter an Input Image of an appropriate format :) ")