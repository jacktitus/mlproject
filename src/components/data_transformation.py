import os
import sys
import numpy as np
import pandas as pd
import src
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class Datatransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.data_transformation_config = Datatransformationconfig()
    
    def get_data_transformation_obj(self):
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy= "median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("impoter",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler())

                ]
            )

            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"(numberical_columns): {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e,sys)
            
    
    def initiate_data_transformation(self,test_data_path,train_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column = "math_score"
            numerical_column =["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]        

            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]        
            
            logging.info("applying the preprocessing in train and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr ,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]
            logging.ingo("saved preprocessing object")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                test_arr,train_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)