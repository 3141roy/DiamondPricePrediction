import streamlit as st
import sys
sys.path.append("..")
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

st.title('Diamond Price Prediction')

carat = st.number_input('Carat')
depth = st.number_input('Depth')
table = st.number_input('Table')
x = st.number_input('X')
y = st.number_input('Y')
z = st.number_input('Z')
cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox('Clarity', ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])

custom_data = CustomData(carat=carat, depth=depth, table=table, x=x, y=y, z=z, cut=cut, color=color, clarity=clarity)

input_data = custom_data.get_data_as_dataframe()

pipeline = PredictPipeline()

if st.button('Predict'):
    predictions = pipeline.predict(input_data)
    st.subheader('Predicted Diamond Price:')
    st.write(predictions)