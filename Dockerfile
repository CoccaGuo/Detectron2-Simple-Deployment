FROM python:3.8.7-slim-buster
LABEL maintainer="CoccaGuo<1927505074@qq.com>"

RUN apt update && apt -y install gcc

# install python ML env. (detectron2 version v0.3)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install Flask \
    && pip install opencv-python-headless \
    && pip install pip install torch==1.7.1+cpu torchvision==0.8.2+cpu --trusted-host download.pytorch.org \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && python -m pip install detectron2 --trusted-host dl.fbaipublicfiles.com\
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html

RUN cd /home && mkdir webapps && cd /home/webapps

WORKDIR /home/webapps
COPY app ./app
COPY engine ./engine
COPY main.py .

EXPOSE 5000
CMD ["python", "main.py"]