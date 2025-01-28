from network_security.src.entities.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from network_security.src.entities.config_entity import DataValidationConfig
from network_security.src.exception.exception import NetworkSecurityException 
from network_security.src.logging.logger import logging 
from network_security.src.constatnts.training_pipeline import SCHEMA_FILE_PATH

from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from network_security.src.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config

            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config['columns'])

            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")

            if len(dataframe.columns)==number_of_columns:
                return True
            
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def check_non_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if there are any non-numerical columns in the given DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to check for non-numerical columns.

        Returns:
            bool: True if non-numerical columns exist, False otherwise.
        """
        try:
            # Get the list of non-numerical columns
            non_numerical_columns = dataframe.select_dtypes(exclude=['number']).columns

            logging.info(f"Non-numerical columns found: {list(non_numerical_columns)}")

            # Return True if there are non-numerical columns, otherwise False
            return len(non_numerical_columns) > 0
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}

            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]

                is_same_dist=ks_2samp(d1,d2)

                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False

                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found                  
                    }})
                
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path,content=report)

            return status
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read the data from train and test
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            # validate number of columns
            train_df_columns_status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not train_df_columns_status:
                logging.info(f"Train dataframe does not contain all columns.")
                
            test_df_columns_status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not test_df_columns_status:
                logging.info(f"Test dataframe does not contain all columns.")   

            # Check for non-numerical columns
            train_df_has_non_numerical_columns = self.check_non_numerical_columns(dataframe=train_dataframe)
            if train_df_has_non_numerical_columns:
                logging.info("The train DataFrame contains non-numerical columns.")

            test_df_has_non_numerical_columns = self.check_non_numerical_columns(dataframe=test_dataframe)
            if test_df_has_non_numerical_columns:
                logging.info("The test DataFrame contains non-numerical columns.")

            validation_status = train_df_columns_status and test_df_columns_status and not train_df_has_non_numerical_columns and not test_df_has_non_numerical_columns

            logging.info(f"Train-df | Number of columns validation: {train_df_columns_status}")
            logging.info(f"Test-df | Number of columns validation: {test_df_columns_status}")
            logging.info(f"Train-df | Non-numerical columns validation: {train_df_has_non_numerical_columns}")
            logging.info(f"Test-df | Non-numerical columns validation: {test_df_has_non_numerical_columns}")

            # lets check data drift
            data_drift_status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)

            logging.info(f"Data drift status: {data_drift_status}")

            if data_drift_status:
                dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(dir_path,exist_ok=True)

                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path, index=False, header=True
                )

                test_dataframe.to_csv(
                    self.data_validation_config.valid_test_file_path, index=False, header=True
                )
            else:
                dir_path=os.path.dirname(self.data_validation_config.invalid_train_file_path)
                os.makedirs(dir_path,exist_ok=True)

                train_dataframe.to_csv(
                    self.data_validation_config.invalid_train_file_path, index=False, header=True
                )

                test_dataframe.to_csv(
                    self.data_validation_config.invalid_test_file_path, index=False, header=True
                )
            
            validation_status = validation_status and data_drift_status

            data_validation_artifact = DataValidationArtifact(
                validation_status = validation_status,
                valid_train_file_path = self.data_ingestion_artifact.train_file_path if validation_status else None,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path if validation_status else None,
                invalid_train_file_path = None if validation_status else self.data_ingestion_artifact.train_file_path,
                invalid_test_file_path = None if validation_status else self.data_ingestion_artifact.train_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)




