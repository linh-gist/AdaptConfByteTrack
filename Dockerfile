# DOCKER COMMANDS
# https://docs.docker.com/engine/install/ubuntu/
# Docker Build: docker build --no-cache -t linhma/adaptbytetrack .
# Remove all unused containers, networks, images, and volumes: docker system df, docker system prune
# Removing images: docker rmi $(docker images -a -q)
# Stop all the containers: docker stop $(docker ps -a -q)
# Remove all the containers: docker rm $(docker ps -a -q)
# Push to share Docker images to the Docker Hub: docker push linhma/adaptbytetrack

# HOW TO USE?
# 1. Running Docker:  docker run -d -p "36901:6901" --name quick --hostname quick linhma/adaptbytetrack
# 2. Openning Browser: http://localhost:36901/vnc.html?password=headless => choose 'noVNC Full Client' => password 'headless'
# 3. Refer: https://accetto.github.io/user-guide-g3/quick-start/

# Use the base image
FROM accetto/ubuntu-vnc-xfce-chromium-g3:20.04

# Switch to root user if necessary
USER root

# Set the working directory
WORKDIR /app

# Download and install Miniconda
# wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-1-Linux-x86_64.sh
COPY ./Miniconda3-py38_23.11.0-1-Linux-x86_64.sh ./
RUN bash Miniconda3-py38_23.11.0-1-Linux-x86_64.sh -b -p /opt/miniconda

# Update PATH
ENV PATH="/opt/miniconda/bin:${PATH}"

# Install build essentials including g++, make
RUN apt-get update && apt-get install -y \
    build-essential g++ make cmake \
    python3-distutils \
    libgl1-mesa-glx \
    libglib2.0-0\
    libeigen3-dev \
    libopencv-dev \
    git

# Initialize Conda
RUN conda init bash
RUN conda install -y python=3.8.0

# Copy your code into the container
COPY ./AdaptConfByteTrack ./AdaptConfByteTrack
COPY ./datasets ./datasets
COPY ./eigen-3.4.0 ./eigen-3.4.0

# Install required Python packages directly in the base Conda environment
RUN pip install "setuptools<65" && \
    pip install numpy==1.23.1 && \
    pip install opencv-python==4.9.0.80 && \
    pip install loguru==0.7.2 && \
    pip install scipy==1.10.1 && \
    pip install lap==0.5.12 && \
    pip install cython_bbox==0.1.5 && \
    pip install matplotlib==3.5.3 && \
    pip install filterpy==1.4.5 && \
    pip install motmetrics==1.4.0 && \
    pip install openpyxl==3.1.5 && \
    pip install pycocotools==2.0.7 && \
    pip install tabulate==0.9.0
RUN git clone https://github.com/JonathonLuiten/TrackEval.git
RUN bash -c "cd /app/TrackEval/ && python setup.py build develop"

# Build c++ packages project
RUN bash -c "cd /app/AdaptConfByteTrack/cppadaptbytetrack/ && python setup.py build develop"

# Activate the environment by default (optional)
RUN echo "source activate base" >> ~/.bashrc

# Specify the command to run your application (if needed)
CMD ["bash"]