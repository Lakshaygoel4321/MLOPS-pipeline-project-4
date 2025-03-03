from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformer,DataTransformerConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig
from src.logger import logging
from src.exception import CustomException
import sys

if __name__ == "__main__":

    try:

        obj = DataIngestion()
        train_array,test_array = obj.initiate_data_ingestion()

        data_transformer = DataTransformer()
        train_file_array,test_file_array,_ = data_transformer.initiate_data_transformer(train_array,test_array)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_file_array,test_file_array)

    except Exception as e:
        raise CustomException(e,sys)
    
