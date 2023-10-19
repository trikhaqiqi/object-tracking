FROM tensorflow/tensorflow:2.5.0

RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["python", "object_tracker.py"]
