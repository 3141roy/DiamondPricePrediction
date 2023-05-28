import streamlit as st
import sys
sys.path.append("..")
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

# Create a Streamlit app
st.title('Diamond Price Prediction')

# Create input fields for user input
carat = st.number_input('Carat')
depth = st.number_input('Depth')
table = st.number_input('Table')
x = st.number_input('X')
y = st.number_input('Y')
z = st.number_input('Z')
cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox('Clarity', ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])

# Create a CustomData instance with user input
custom_data = CustomData(carat=carat, depth=depth, table=table, x=x, y=y, z=z, cut=cut, color=color, clarity=clarity)

# Get the input data as a DataFrame
input_data = custom_data.get_data_as_dataframe()

# Create an instance of the prediction pipeline
pipeline = PredictPipeline()

# Check if user has provided input
if st.button('Predict'):
    # Call the predict method with the input data
    predictions = pipeline.predict(input_data)

    # Display the predictions
    st.subheader('Predicted Diamond Price:')
    st.write(predictions)
