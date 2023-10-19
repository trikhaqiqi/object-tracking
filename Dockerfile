FROM tensorflow/tensorflow:2.5.0

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN \
	apt-get install -y \
	wget \
	unzip \
	ffmpeg \ 
	git

COPY . /home/abandoned-yolo

WORKDIR /home/abandoned-yolo

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip install -r requirements.txt

# RUN wget https://pjreddie.com/media/files/yolov3.weights -P weights/
RUN python load_weights.py --weights ./weights/yolov3.weights --output ./weights/yolov3.tf

CMD ["python", "object_tracker.py", "--video", "./data/video/macet1.mp4", "--output", "./data/video/macet1.avi"]