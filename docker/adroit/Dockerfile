# Make sure to create the rfcl_base image from the base folder

FROM rfcl_base:latest
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN pip install minari && pip install gymnasium-robotics && pip install gdown
