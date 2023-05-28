import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer #To handle missing values
from sklearn.preprocessing import StandardScaler #To handle feature scaling
from sklearn.preprocessing import OrdinalEncoder #To handle encoding of Ordinal data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

#Data Transformation Config

@dataclass
class DataTransformationConfig:
    #We give path of pickle here
    preprocessor_obj_filepath = os.path.join('artifacts','preprocessor.pkl')


#Data Transformation Config class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        # Make the pickle file
        try:
            logging.info('Data Transformation initiated')

            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']

            cut_ranking = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_ranking = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_ranking = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initiated')

            #Numerical Pipleine
            numPipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            #Categorical Pipeline
            catPipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_ranking,color_ranking,clarity_ranking])),
                    ('scaler',StandardScaler())
                ]
            )

            #Combining the two to pre-process our entire data
            preprocessor = ColumnTransformer(
                [
                    ('numPipeline',numPipeline,numerical_cols),
                    ('catPipeline',catPipeline,categorical_cols)
                ]
            )

            return preprocessor
            
            logging.info('Pipeline Completed')


        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Test and Train datafile read')

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            #Dividing into independent and dependent
            input_features_train = train_df.drop(columns = drop_columns,axis=1)
            target_features_train = train_df[target_column_name]

            input_features_test = test_df.drop(columns = drop_columns,axis=1)
            target_features_test = test_df[target_column_name]

            # Performing the transformation

            train_input_arr = preprocessing_obj.fit_transform(input_features_train)
            test_input_arr = preprocessing_obj.transform(input_features_test)

            logging.info('Applied preprocssing on training and test datasets')

            train_arr = np.c_[train_input_arr,np.array(target_features_train)]
            test_arr = np.c_[test_input_arr,np.array(target_features_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)