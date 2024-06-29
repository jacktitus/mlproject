import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class data_ingestion_config:
    train_data_path :str = os.path.join('artifacts',"train.csv")
    test_data_path  :str = os.path.join('artifacts',"test.csv")
    raw_data_path :str = os.path.join('artifacts',"data.csv")

class data_ingestion:
    def __init__(self):
        self.ingestion_config = data_ingestion_config()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs (os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header= True)

            logging.info("train test split initiated")

            train_set,test_set = train_test_split(df,test_size=0.2 , random_state= 42)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header= True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header= True)
            logging.info("ingestion is completed")

            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path # return for data transformatiom
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    obj = data_ingestion()
    obj.initiate_data_ingestion()