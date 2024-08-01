FROM continuumio/miniconda3

# Install sudo and vim
RUN apt-get update && apt-get install -y sudo vim

# Copy the environment.yml file into the container
COPY environment.yml /tmp/environment.yml

# Create the environment
RUN conda env create -f /tmp/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "Emotiv", "/bin/bash", "-c"]

# Ensure the environment is activated when the container starts
ENTRYPOINT ["conda", "run", "-n", "Emotiv", "/bin/bash", "-c"]

# Copy your application's source code into the container
COPY . /FastSpeech2

# Set the working directory to /FastSpeech2
WORKDIR /FastSpeech2

