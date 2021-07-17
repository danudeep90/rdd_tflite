
import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict
import pandas as pd


# Normalize co-ordinates
def normalize_coordinates(imgFilePath, minX, minY, maxX, maxY):

    # Extracting size of image
    image = Image.open(imgFilePath)
    w = int(image.size[0])
    h = int(image.size[1])

    dw = 1./w
    dh = 1./w

    # Multiplying with actual co-ordinates to standardize
    minX_norm = minX*dw
    minY_norm = minY*dh
    maxX_norm = maxX*dw
    maxY_norm = maxY*dh

    return [minX_norm, minY_norm, maxX_norm, maxY_norm]


def main(input_train_file, input_valid_file, input_test_file, class_file):

    print("Preparing training dataset csv")

    # dictionary to store list of image paths in each class
    imageListDict = defaultdict(set)

    # damage classes info
    f = open(class_file, "r")
    classes_info = f.readlines()
    f.close()

    # Read the file
    with open(input_train_file, "r") as f:
        lst_img_details_train = f.readlines()
    f.close()

    with open(input_valid_file, "r") as f:
        lst_img_details_valid = f.readlines()
    f.close()

    with open(input_test_file, "r") as f:
        lst_img_details_test = f.readlines()
    f.close()

    # Combining both the lists
    img_details_dict = {"TRAINING":lst_img_details_train,
                        "VALIDATION":lst_img_details_valid,
                        "TEST":lst_img_details_test}

    # Declare an empty list to store details of labels
    lst_labels_info = []

    # Loop over all types of data
    for key, value in img_details_dict.items():

        # dataset_type
        ds_type = key
        lst_img_details = value

        # Loop over each file and create a labels related info as well
        for imgFilePath in lst_img_details:

            # Stripping the newline character
            imgFilePath = imgFilePath.strip()

            # Getting image full file path
            imgFileFullPath = os.path.abspath(imgFilePath)

            # Removing \n at the end of filename and extract info without .jpg extension
            tmp_str1 = imgFilePath.strip().rsplit(".", 1)[0]

            # Extracting the actual XML path of the image annotation
            actual_xml_path = tmp_str1.replace("images", "annotations/xmls") + ".xml"
            actual_xml_fullpath = os.path.abspath(actual_xml_path)

            # Parsing the XML file to convert to yolo annotation
            tree = ET.parse(actual_xml_fullpath)
            root = tree.getroot()

            # loop over each object tag in annotation tag
            for objects in root.findall('object'):
                defect_type = objects.find('name').text.replace(" ", "")

                if defect_type == "D00" or defect_type == "D10" or defect_type == "D20" or defect_type == "D40":
                    bndbox = objects.find('bndbox')
                    [minX, minY, maxX, maxY] = [int(child.text) for child in bndbox]

                    # Convert co-ordinates to range [0,1]
                    [minX_norm, minY_norm, maxX_norm, maxY_norm] = normalize_coordinates(imgFilePath, minX, minY, maxX, maxY)

                    # Add info to the a temp dict
                    tmp_dict_label_info = {"dataset_type":ds_type,
                                          "file_path":imgFileFullPath,
                                          "defect_type":defect_type,
                                          "minX":minX_norm,
                                          "minY":minY_norm,
                                          "dummy1":"",
                                          "dummy2":"",
                                          "maxX":maxX_norm,
                                          "maxY":maxY_norm,
                                          "dummy3":"",
                                          "dummy4":""}

                    # Adding this to the list
                    lst_labels_info.append(tmp_dict_label_info)

                    imageListDict[defect_type].add(imgFilePath)
    
    # Create a dataframe and writing it
    train_df = pd.DataFrame(lst_labels_info)

    # Printing summary of the datasets
    print("------- Summary of images in the dataset---")
    for key,value in img_details_dict.items(): 
        print("# of",key,"images", ":", len(value))
    print("-------------------------------------------")

    # Printing label info of the datasets
    #Printing how many of each class are available
    print("----- Summary of labels in the dataset-----")
    for cl in imageListDict:
        print(cl.strip(), ":", len(imageListDict[cl]))
    print("-------------------------------------------")

    # Write as output
    output_csv = "rdd_train_dataset.csv"
    print("Writing", output_csv, "to", os.getcwd())
    train_df.to_csv(output_csv, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Code to convert xml files to yolo format")
    parser.add_argument('--input_train_file', type=str,
                        help='Path of txt file which contains details of images for training',
                        default="datasets/train.txt")
    parser.add_argument('--input_valid_file', type=str,
                        help='Path of txt file which contains details of images for validation',
                        default="datasets/valid.txt")
    parser.add_argument('--input_test_file', type=str,
                    help='Path of txt file which contains details of images for test',
                    default="datasets/test.txt")
    parser.add_argument('--class_file', type=str,
                        help='Path of txt file which has classes information',
                        default="damage_classes.txt")

    args = parser.parse_args()

    input_train_file = args.input_train_file
    input_valid_file = args.input_valid_file
    input_test_file = args.input_test_file
    class_file = args.class_file

    # Function call
    main(input_train_file, input_valid_file, input_test_file, class_file)