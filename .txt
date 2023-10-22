# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# MAINTAINER ilyassstri 

# RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev


# RUN \
# 	apt-get install -y \
# 	wget \
# 	unzip \
# 	ffmpeg \ 
# 	git

# COPY requirements.txt .

# RUN pip3 install -r requirements.txt

# WORKDIR home/

# # RUN git clone https://github.com/pjreddie/darknet
# # WORKDIR darknet/

# # RUN sed -i 's/GPU=.*/GPU=1/' Makefile 
# # RUN sed -i 's/CUDNN=.*/CUDNN=1/' Makefile && \
# # 	make

# RUN wget https://pjreddie.com/media/files/yolov3.weights -P weights/

# COPY ./data/video/test.mp4 /app/data/video/

# WORKDIR /home

# CMD ["python", "object_tracker.py", "--video", "./data/video/test.mp4", "--output", "./data/video/results.avi"]

FROM python:3
RUN \
 apt-get install -y \
 wget \
 unzip \
 git

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
RUN pip3 install numpy
# RUN pip3 install tensorflow
RUN pip3 install lxml
RUN pip3 install tqdm
RUN pip3 install seaborn
RUN pip3 install pillow
RUN pip3 install opencv-python
RUN pip3 install requests
# RUN pip3 install numba
RUN pip3 install imutils

WORKDIR home/abandoned-yolo/
COPY object_tracker.py /home/abandoned-yolo
# COPY test_img.jpg /home/
# RUN git clone https://github.com/pjreddie/darknet

WORKDIR abandoned-yolo/
RUN wget https://pjreddie.com/media/files/yolov3.weights -P weights/
WORKDIR /home/abandoned-yolo

CMD ["python", "object_tracker.py", "--video", "./data/video/macet1.mp4", "--output", "./data/video/macet1.avi"]
