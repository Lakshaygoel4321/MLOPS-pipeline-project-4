from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_object
import os
import pandas as pd
import numpy as np
import logging
import sys

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()
    
    def get_data_transformerd(self):
        try:

            numerical_column = ['reading_score','writing_score']
            categorical_column = [
                    'gender',
                    'race_ethnicity',
                    'parental_level_of_education',
                    'lunch',
                    'test_preparation_course'	
            
            ]

            num_feature = Pipeline(steps=[
                ('num_feature',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_feature = Pipeline(steps=[
                ('categorical feature',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'categorical column: {categorical_column}')
            logging.info(f"numerical column: {numerical_column}")

            preprocessor = ColumnTransformer([
                ('numerical_fea',num_feature,numerical_column),
                ('categorical_fea',cat_feature,categorical_column)
            ])


            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_data_transformer(self,train_path,test_path):
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessor_obj = self.get_data_transformerd()

            target_column_name = 'math_score'
            numerical_column = ['reading_score','writing_score']

            # train
            input_feature_train_df = train_df.drop([target_column_name],axis=1)
            input_target_train_df = train_df[target_column_name]

            # test
            input_feature_test_df = test_df.drop([target_column_name],axis=1)
            input_target_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing on training and testing data')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(input_target_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(input_target_test_df)
            ]

            logging.info('Saved preprocessing obj')

            save_object(
                file_path = self.data_transformer_config.preprocessor_obj_file,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file

            )  
        
        except Exception as e:
            raise CustomException(e,sys)
        
