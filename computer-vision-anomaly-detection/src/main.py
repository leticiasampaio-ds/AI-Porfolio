import os
from pathlib import Path
from preprocessing.image_preprocessor import ImagePreprocessor

class Main:
    """
    Entry point for running data preprocessing and model training pipeline.

    Args:
        preprocess_data_flag (bool): Flag to run the preprocess pipeline or not.
    """
    def __init__(self, preprocess_data_flag: bool):
        self.preprocess_data_flag = preprocess_data_flag

    def preprocess_images(self):
        train_input_raw_data_path = Path('/home/leticia/projetos/AI-Porfolio/computer-vision-anomaly-detection/data/raw/train/images')
        valid_input_raw_data_path = Path('/home/leticia/projetos/AI-Porfolio/computer-vision-anomaly-detection/data/raw/valid/images')
        output_dir_path = Path('/home/leticia/projetos/AI-Porfolio/computer-vision-anomaly-detection/data/processed')
        ImagePreprocessor(train_input_raw_data_path, output_dir_path).run()
        ImagePreprocessor(valid_input_raw_data_path, output_dir_path).run()

    def run_model_pipeline(self):
        pass

    def run(self):
        if self.preprocess_data_flag == True:
            self.preprocess_images()
        else:
            self.run_model_pipeline()

if __name__ == "__main__":
    app = Main(True)
    app.run()