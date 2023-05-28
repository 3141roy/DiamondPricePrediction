import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
if __name__=='__main__':
    obj = DataIngestion()
    train_datapath,test_datapath = obj.initiate_data_ingestion()
    print(train_datapath,test_datapath)

    data_transformation = DataTransformation()

    train_arr,test_arr,obj_path = data_transformation.initiate_data_transformation(train_datapath,test_datapath)

    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)