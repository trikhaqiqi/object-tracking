FROM python:3.7.17

RUN apt-get update && apt-get install -y \
    xvfb x11-xserver-utils

WORKDIR /home/app

COPY . /home/app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["xvfb-run", "--server-args", "-screen 0 1920x1080x24", "python", "stuck_on_the_road.py"]