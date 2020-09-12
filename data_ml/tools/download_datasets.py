# convenience script to download, unzip test/train data from the podcast data archive
# combine the train data and test data into a different dataset
import argparse
import os
import sys
import subprocess
import shutil
import gzip
import tarfile
import pandas as pd


def aws_download(s3_path, local_dir):
    command = "aws --no-sign-request s3 cp " + s3_path + " " + local_dir
    os.system(command)

    filename = os.path.basename(s3_path)
    full_path_to_filename = os.path.join(local_dir, filename)
    if not os.path.exists(full_path_to_filename):
        raise RuntimeError("{} does not exist".format(full_path_to_filename))
    return full_path_to_filename

def unzip_and_extract(extract_dir, dataset_archive, clean_archive=True):
    # unzip file to folder
    # delete zip file after extraction
    extracted_folder_name = os.path.basename(dataset_archive).split(".tar.gz")[0]
    extracted_folder_path = os.path.join(extract_dir, extracted_folder_name)
    if dataset_archive.endswith("tar.gz"):
        tar = tarfile.open(dataset_archive, "r:gz")
        tar.extractall(path=extracted_folder_path)
        tar.close()

    if clean_archive:
        os.remove(dataset_archive)

    return extracted_folder_path

def download_unzip_and_combine(test_dataset_paths, test_data_download_location, is_test = True):

    if os.path.exists(test_data_download_location):
        shutil.rmtree(test_data_download_location)
    os.mkdir(test_data_download_location)

    local_extracted_locations = []
    for test_data in test_dataset_paths:
        local_download_location = aws_download(test_data, test_data_download_location)
        local_extracted_location = unzip_and_extract(test_data_download_location, local_download_location)
        local_extracted_locations.append(local_extracted_location)

    # create a wav directory
    wav_dir = os.path.join(test_data_download_location, "wav")
    os.mkdir(wav_dir)

    if is_test:
        tsv_file = os.path.join(test_data_download_location, "test.tsv")
    else:
        tsv_file = os.path.join(test_data_download_location, "train.tsv")
        
    dev_tsv_file = os.path.join(test_data_download_location, "dev.tsv")
    with open(tsv_file, "w") as outfile:
        for folder_num in range(0, len(local_extracted_locations)):
            extracted_location = local_extracted_locations[folder_num]

            inner_folder = os.listdir(extracted_location)[0]
            sub_wav_dir = os.path.join(extracted_location, inner_folder, "wav")
            for filename in os.listdir(sub_wav_dir):
                full_filename_path = os.path.join(sub_wav_dir, filename)
                # copy all wav files from sub_wav_dir to wav_dir
                shutil.copy(full_filename_path, wav_dir)
            
            if is_test:
                sub_test_tsv = os.path.join(extracted_location, inner_folder, "test.tsv")
            else:
                sub_test_tsv = os.path.join(extracted_location, inner_folder, "train.tsv")

            # for all but the first folder, skip the header line and only pick out the first 6 columns
            df = pd.read_csv(sub_test_tsv,sep='\t')
            relevant_columns = df.iloc[:, :6]
            relevant_columns.to_csv(outfile, sep='\t', header=(folder_num==0))

            # write validation dataset
            if not is_test:
                sub_dev_tsv = os.path.join(extracted_location, inner_folder, "dev.tsv")
                if os.path.exists(sub_dev_tsv):
                    df_dev = pd.read_csv(sub_dev_tsv,sep='\t')
                    df_dev_columns = df_dev.iloc[:, :6]
                    with open(dev_tsv_file, "a") as combined_dev_file:
                        df_dev_columns.to_csv(combined_dev_file, sep='\t', header=(folder_num==0))

            # delete extracted location
            shutil.rmtree(extracted_location)

    return test_data_download_location


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Orcasound datasets. Prerequisite: Ensure awscli is setup & you can run 'aws s3 help'")
    parser.add_argument("download_dir", type=str, help="Directory to save the extracted dataset(s)")
    # parser.add_argument("dataset_type", type=str, help="Specify either (train/test)")
    parser.add_argument("--only_train", default=False, action='store_true')
    parser.add_argument("--only_test", default=False, action='store_true')
    args = parser.parse_args()

    combined_train_dataset = "s3://acoustic-sandbox/labeled-data/detection/train/TrainDataLatest_PodCastAllRounds.tar.gz"   
    test_datasets = [
        "s3://acoustic-sandbox/labeled-data/detection/test/OS_SVeirs_07_05_2019_08_24_00.tar.gz",
        "s3://acoustic-sandbox/labeled-data/detection/test/OrcasoundLab09272017_Test.tar.gz"
    ]
    # individual_train_datasets = [
    #     "s3://acoustic-sandbox/labeled-data/detection/train/WHOIS09222019_PodCastRound1.tar.gz",
    #     "s3://acoustic-sandbox/labeled-data/detection/train/OrcasoundLab07052019_PodCastRound2.tar.gz",
    #     "s3://acoustic-sandbox/labeled-data/detection/train/OrcasoundLab09272017_PodCastRound3.tar.gz"
    # ]

    if not args.only_train:
        final_path = download_unzip_and_combine(
            test_datasets, os.path.join(args.download_dir, "TestDataLatest_PodCastAllRounds")
            )
        print("Test data extracted to {}".format(final_path))

    if not args.only_test:
        dataset_archive = aws_download(combined_train_dataset, args.download_dir)
        final_path = unzip_and_extract(args.download_dir, dataset_archive)
        print("Train data extracted to {}".format(final_path))

