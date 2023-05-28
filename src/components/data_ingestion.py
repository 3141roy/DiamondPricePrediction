import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

#Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_datapath = os.path.join('artifacts','train.csv')
    test_datapath = os.path.join('artifacts','test.csv')
    raw_datapath = os.path.join('artifacts','raw.csv')


#Create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method Starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data','data.csv'))
            logging.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_datapath),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_datapath,index=False)
            logging.info("Data saved as raw data path")

            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)
            logging.info('Splitting into training and testing set done')

            train_set.to_csv(self.ingestion_config.train_datapath,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_datapath,index=False,header=True)
            logging.info('Test and Train datasets saved in artifacts')

            return(
                self.ingestion_config.train_datapath,
                self.ingestion_config.test_datapath
            )

        except Exception as e:
            logging.info('Error occured in Data Ingestion Config')
