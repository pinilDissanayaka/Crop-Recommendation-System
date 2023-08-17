import streamlit as st
import numpy as np
import pickle

with open('../Notebook/predictor.pickle', 'rb') as file:
    model = pickle.load(file)


st.title("Crop recommendation system")


def prediction(data):
    return model.predict(data)
    


def main():

    n = st.slider("Nitrogen level : ", 0, 280, 1)
    st.write("Nitrogen level  = ", n)

    p = st.slider("Posparas level : ", 0, 280, 1)
    st.write("Posparas level  = ", p)

    k = st.slider("Potassium level : ", 0, 280, 1)
    st.write("Potassium level  = ", k)

    temp = st.slider("Tempurature  (C): ", 0, 60, 1)
    st.write("Tempurature level  = ", temp)

    hum = st.slider("Humidity : ", 0, 150, 1)
    st.write("Humidity level  = ", hum)

    ph = st.slider("pH : ", 0, 14, 1)
    st.write("pH level  = ", ph)

    rf = st.slider("Rainfall : ", 0, 300, 1)
    st.write("Rainfall level  = ", rf)
    
    input_data = [n, p, k, temp, hum, ph, rf]
    input_data_arr = np.asarray(input_data).reshape(1, -1)
    
   
    

    if st.button("Predict"):
        pred = prediction(input_data_arr)
        st.write("Predicted crop is ")
        st.success(pred[0])


if __name__ == '__main__' :
    main()