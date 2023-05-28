
# Diamond Price Prediction

LINK : https://3141roy-diamondpriceprediction.streamlit.app/

The goal is to predict `price` of given diamond (Regression Analysis).

There are 10 independent variables (including `id`):

* `id` : unique identifier of each diamond
* `carat` : Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
* `cut` : Quality of Diamond Cut
* `color` : Color of Diamond
* `clarity` : Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these characteristics under 10-power magnification.
* `depth` : The depth of diamond is its height (in millimeters) measured from the culet (bottom tip) to the table (flat, top surface)
* `table` : A diamond's table is the facet which can be seen when the stone is viewed face up.
* `x` : Diamond X dimension
* `y` : Diamond Y dimension
* `x` : Diamond Z dimension

Target variable:
* `price`: Price of the given Diamond.

The data contains oridnal features which require domain knowledge to interpret.

For more information about these ordinal features, 
[Click Here](https://www.americangemsociety.org/ags-diamond-grading-system/)


[Source of Dataset](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)




## Running the Code

To deploy this project run

`Step 1` : Create a virtual env and activate it 

`Step 2` : Install the dependencies by running 

            pip install -r requirements.txt

`Step 3` : The project is hosted on Streamlit but to run locally, you can use the command : streamlit run app\app.py

## Documentation

This project involves a step by step process of performing all operations that a Machine Learning project requires in a step by step fashion. 

We first begin with EDA (Exploratory Data Analysis) and Model Analysis, both of which can be seen in the folder `notebooks` under `EDA.ipynb` and `Model.ipynb`. The dataset used is stored in ` notebooks\data `

Apart from this, we also use a Custom Exception Handler to ensure that the code does not stop unexpectedly during execution and a Logger to to help during debugging process. Both these are present in the `src` folder itself under the name `exception.py` and `logger.py`.

The `utils.py` is another file in the `src` folder. It is a utility python executable that has three functions
1. `save_object` : This functuon is passed with two parameters, the path and the file name and is required to store the files generated during model creation i.e. the pickle files for pre-processing our input and the pickle file for the model itself at the desired location.

2. `evaluate_model` : This function is used to evaluate and generate report by testing our data against a list of models passed as a parameter to it.

3. `load_object` : This function is used to load the files from a file path during execution phase when the model is being run

Apart from this we also have a `pipeline` folder and `components` folder to automate our processes


