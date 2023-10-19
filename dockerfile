FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER ilyassstri 

RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev


RUN \
	apt-get install -y \
	wget \
	unzip \
	ffmpeg \ 
	git

COPY requirements.txt .

RUN pip3 install -r requirements.txt

WORKDIR home/

# RUN git clone https://github.com/pjreddie/darknet
# WORKDIR darknet/

# RUN sed -i 's/GPU=.*/GPU=1/' Makefile 
# RUN sed -i 's/CUDNN=.*/CUDNN=1/' Makefile && \
# 	make

RUN wget https://pjreddie.com/media/files/yolov3.weights -P weights/

COPY ./data/video/test.mp4 /app/data/video/

WORKDIR /home

CMD ["python", "object_tracker.py", "--video", "./data/video/test.mp4", "--output", "./data/video/results.avi"]
