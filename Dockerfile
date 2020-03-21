FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y screen

RUN mkdir -p /workspace/mighty
COPY . /workspace/mighty/
RUN pip install -e /workspace/mighty/

ENV VISDOM_PORT 8098

CMD screen -dmS visdom visdom -port $VISDOM_PORT ; /bin/bash