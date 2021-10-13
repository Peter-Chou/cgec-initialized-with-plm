FROM ubuntu:18.04

# use huawei mirror
RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list && \
	sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

RUN  apt-get update && apt-get install -y \
	tzdata \
	unzip \
	curl \
	wget \
	git \
	vim \
	python-pip \
	python3-pip \
	&& \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone -b version3.2 https://github.com/nusnlp/m2scorer.git
RUN pip --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple \
	nltk==3.4

RUN wget http://59.108.48.37:9014/lcwm/pkunlp/downloads/libgrass-ui.tar.gz
RUN tar -xvf libgrass-ui.tar.gz && rm -rf /opt/libgrass-ui.tar.gz

# download nlpcc 2018 gold
RUN git clone https://github.com/zhaoyyoo/NLPCC2018_GEC.git
WORKDIR /opt/NLPCC2018_GEC
RUN git checkout cd9fb2064f9026c5c8bbae0c654581e6ee071904
RUN unzip Data.zip && mv Data/test/gold /opt

WORKDIR /opt
RUN rm -rf /opt/NLPCC2018_GEC
COPY ./segment_test_file.py  /opt/segment_test_file.py
COPY ./entrypoint.sh  /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
