#!/bin/bash

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null
then
    echo "AWS CLI is not installed. Please install it and configure it before running this script."
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null
then
    echo "AWS CLI is not configured. Please configure it before running this script."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "jq is not installed. Please install it before running this script."
    exit 1
fi

# Define variables
IMAGE_NAME="fastspeech2"
S3_BUCKET="rosettaartifacts"
S3_PATH="builds/${IMAGE_NAME}"

# List the most recent 20 versions in S3 sorted by newest first
echo "Available versions in S3 (most recent 20):"
aws s3 ls "s3://${S3_BUCKET}/${S3_PATH}/" --recursive | sort -r | head -n 20 | awk '{print $4}' | sed 's/.*_\(.*\)\.tar/\1/' | tr '-' '.' | sort -rV | head -n 20

# Prompt the user to select a version
read -p "Enter the version you want to download: " VERSION
VERSION=${VERSION//./-}
TAR_FILE="${IMAGE_NAME}_${VERSION}.tar"

# Check if the tar file already exists locally
if [[ -f "${TAR_FILE}" ]]; then
    read -p "File ${TAR_FILE} already exists locally. Do you want to overwrite it? (y/n) " OVERWRITE
    if [[ "$OVERWRITE" != "y" ]]; then
        echo "Using existing local file."
    else
        echo "Downloading ${TAR_FILE} from S3..."
        aws s3 cp "s3://${S3_BUCKET}/${S3_PATH}/${TAR_FILE}" .
    fi
else
    echo "Downloading ${TAR_FILE} from S3..."
    aws s3 cp "s3://${S3_BUCKET}/${S3_PATH}/${TAR_FILE}" .
fi

# Load the Docker image from the tar file
docker load -i "${TAR_FILE}"

echo "Docker image ${IMAGE_NAME}:${VERSION} loaded successfully from ${TAR_FILE}"

# ask if the user wants to delete the tar file
read -p "Do you want to delete the tar file? (y/n) " DELETE
if [[ "$DELETE" == "y" ]]; then
    rm "${TAR_FILE}"
    echo "Deleted ${TAR_FILE}"
fi