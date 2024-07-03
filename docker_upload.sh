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

# Read version from package.json
VERSION=$(jq -r .version < ./package.json)

# Replace dots with hyphens in the version
VERSION=${VERSION//./-}

# Confirm the version to be uploaded
read -p "Are you sure you want to upload version ${VERSION}? (y/n) " CONFIRM
if [[ "$CONFIRM" != "y" ]]; then
    echo "Aborted by user."
    exit 1
fi

# Define variables
IMAGE_NAME="fastspeech2"
TAR_FILE="${IMAGE_NAME}_${VERSION}.tar"
S3_BUCKET="rosettaartifacts"
S3_PATH="builds/${IMAGE_NAME}/${TAR_FILE}"

# Check if the version already exists in S3
if aws s3 ls "s3://${S3_BUCKET}/${S3_PATH}" &> /dev/null; then
    read -p "Version ${VERSION} already exists in S3. Do you want to overwrite it? (y/n) " OVERWRITE
    if [[ "$OVERWRITE" != "y" ]]; then
        echo "Aborted by user."
        exit 1
    fi
fi

# Save the Docker image to a tar file
docker save -o "${TAR_FILE}" "${IMAGE_NAME}"

# Upload the tar file to AWS S3
aws s3 cp "${TAR_FILE}" "s3://${S3_BUCKET}/${S3_PATH}"

echo "Docker image saved to ${TAR_FILE} and uploaded to s3://${S3_BUCKET}/${S3_PATH}"