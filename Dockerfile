# FROM tensorflow/tensorflow:2.5.0
# FROM tensorflow/tensorflow:2.5.0-jupyter
FROM python:3.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
	unzip \
	ffmpeg \ 
	git

# ENV CUDA_VISIBLE_DEVICES=-1
ENV CUDA_VISIBLE_DEVICES=0
# ENV DISPLAY=:0

# COPY . /home/abandoned-yolo

# WORKDIR /home/abandoned-yolo

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip

# RUN pip install -r requirements.txt

RUN pip3 install tensorflow
RUN pip3 install numpy
RUN pip3 install opencv-python
RUN pip3 install lxml
RUN pip3 install tqdm
RUN pip3 install seaborn
RUN pip3 install pillow

# RUN wget https://pjreddie.com/media/files/yolov3.weights -P weights/
# RUN python load_weights.py --weights ./weights/yolov3.weights --output ./weights/yolov3.tf

CMD ["python", "object_tracker.py", "--video", "./data/video/macet1.mp4", "--output", "./data/video/macet1.avi"]