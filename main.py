from network_security.src.components.data_ingestion import DataIngestion

from network_security.src.exception.exception import NetworkSecurityException
from network_security.src.logging.logger import logging
from network_security.src.entities.config_entity import DataIngestionConfig
from network_security.src.entities.config_entity import TrainingPipelineConfig

import sys

if __name__=='__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiate the data ingestion")

        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logging.info("Data Initiation Completed")
        
        print(data_ingestion_artifact)

        
    except Exception as e:
        raise NetworkSecurityException(e,sys)
