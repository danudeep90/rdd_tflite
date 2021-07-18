import numpy as np
import os
import argparse

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

from datetime import datetime
from pytz import timezone

# tf.get_logger().setLevel('ERROR')
# from absl import logging
# logging.set_verbosity(logging.ERROR)

import logging, sys


def main(train_csv_path, num_epochs):

    # Choosing a model architecture
    spec = model_spec.get('efficientdet_lite4')

    # Loading the dataset
    train_data, \
    validation_data,\
    test_data = object_detector.DataLoader\
            .from_csv(train_csv_path)

    # Train the model
    model = object_detector.create(train_data, model_spec=spec, batch_size=8,
                                train_whole_model=True,
                                validation_data=validation_data,
                                epochs=num_epochs)

    # Test the model
    model.evaluate(test_data)
    
    # Exporting the model
    model.export(export_dir='.')

    # Evaluate the tensorflow lite model
    model.evaluate_tflite('model.tflite', test_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Code to train tf lite model")
    parser.add_argument('--train_csv_path', type=str,
                        help='Path of csv file which contains details of images, bounding boxes for training',
                        default="rdd_train_dataset.csv")
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs',
                        default=1)

    args = parser.parse_args()

    train_csv_path = args.train_csv_path
    num_epochs = args.num_epochs

    # Create a logging directory if it doesnt exist
    logs_dirpath = os.path.join(os.getcwd(),"logs")
    if not os.path.exists(logs_dirpath):
        os.mkdir(logs_dirpath)

    # Log file name
    # Creating datetime type folders
    ist_tz = timezone('Asia/Kolkata')
    india_datetime = datetime.now(ist_tz)
    file_prefix_str = india_datetime.strftime("%d_%b_%Y-%H_%M_%S")
    log_file_name = "log"+"_"+file_prefix_str

    # Setup logger
    log_file_full_path = os.path.join(logs_dirpath, log_file_name + ".log")
    logging.basicConfig(filename=log_file_full_path, level=logging.DEBUG)
    logger = logging.getLogger()
    sys.stderr.write = logger.error
    sys.stdout.write = logger.info

    # Print message 
    print("Starting training using {} for {} epochs".format(train_csv_path, num_epochs))
 
    # Function call
    main(train_csv_path, num_epochs)