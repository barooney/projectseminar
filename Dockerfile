FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install reverse_geocoder nltk tqdm matplotlib pandas sklearn numpy

VOLUME /project
WORKDIR /project

CMD ["python3"]