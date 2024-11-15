import boto3
import os
import subprocess
import argparse

# Define S3 bucket and folder name
bucket_name = 'athvaith-08'
folder_name = 'lightning_session'

# Get the latest git commit ID
def get_git_commit_id():
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_id
    except subprocess.CalledProcessError as e:
        print("Error getting git commit ID:", e)
        return None

# Upload model to S3
def upload_model_to_s3(folder_path):
    commit_id = get_git_commit_id()
    if commit_id is None:
        return

    s3_client = boto3.client('s3')
    for model_file in os.listdir(folder_path):
        model_file_path = os.path.join(folder_path, model_file)
        if os.path.isfile(model_file_path):  # Check if it's a file
            s3_key = f"{folder_name}/{commit_id}/{model_file}"
            try:
                s3_client.upload_file(model_file_path, bucket_name, s3_key)
                print(f"Model uploaded to s3://{bucket_name}/{s3_key}")
            except Exception as e:
                print("Error uploading model to S3:", e)

# Main function to parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload model files from a folder to S3 with git commit ID.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing model files to upload')
    args = parser.parse_args()

    upload_model_to_s3(args.folder_path)
