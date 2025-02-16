from network_security.src.components.data_ingestion import DataIngestion
from network_security.src.components.data_validation import DataValidation
from network_security.src.components.data_transformation import DataTransformation
from network_security.src.components.model_trainer import ModelTrainer

from network_security.src.exception.exception import NetworkSecurityException
from network_security.src.logging.logger import logging
from network_security.src.entities.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from network_security.src.entities.config_entity import TrainingPipelineConfig

import sys

if __name__=='__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()

        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the Data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")        
        print(data_ingestion_artifact)

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("Initiate the Data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")        
        print(data_validation_artifact)
        
        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Initiate the Data Transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("data Transformation completed")
        print(data_transformation_artifact)

        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        logging.info("Model Training started")
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        logging.info("Model Training artifact created")
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys)
