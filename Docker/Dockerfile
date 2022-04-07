FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ARG RELEASE_VERSION="1.0.2"

RUN apt-get update && \
	apt-get install -y git \
	libx11-6 \
	libgl1 \
	libopengl0 \
	libegl1 \
	wget\
	-y unzip

RUN mkdir /app

WORKDIR /app

RUN wget https://github.com/baptistebaquero/ALIDDM/archive/refs/tags/v$RELEASE_VERSION.zip
RUN unzip v$RELEASE_VERSION.zip

RUN conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN pip install -r /app/ALIDDM-$RELEASE_VERSION/requirements.txt


RUN mv ALIDDM-$RELEASE_VERSION/py py
RUN rm -rf v$RELEASE_VERSION.zip \
	rm -rf v$ALIDDM-$RELEASE_VERSION


RUN mkdir /app/model
WORKDIR /app/model
RUN wget https://github.com/baptistebaquero/ALIDDM/releases/download/v$RELEASE_VERSION/models.zip
RUN unzip models.zip

RUN rm -rf models.zip
