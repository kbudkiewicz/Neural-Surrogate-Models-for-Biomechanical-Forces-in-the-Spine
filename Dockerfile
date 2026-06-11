FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime
LABEL authors="krystian_jonas", description="Master Thesis: Neural Surrogate Models for Biomechanical Forces in the Spine"

WORKDIR /src
EXPOSE 2001
RUN apt update
RUN apt -y install nano
RUN apt -y install tmux

COPY . .
RUN pip install -r requirements.txt
RUN nvidia-smi
