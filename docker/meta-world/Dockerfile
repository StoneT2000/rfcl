# Make sure to create the rfcl_base image from the base folder

FROM rfcl_base:latest
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN ln -s /usr/lib/nvidia /usr/lib/nvidia-430 && mkdir -p ~/.mujoco/mujoco200 && wget https://www.roboti.us/download/mujoco200_linux.zip && unzip mujoco200_linux.zip -d ~/.mujoco/ && rm mujoco200_linux.zip && mkdir -p ~/.mujoco/mujoco200 && mv ~/.mujoco/mujoco200_linux/* ~/.mujoco/mujoco200
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/ && rm mujoco210-linux-x86_64.tar.gz
RUN pip install minari && pip install gymnasium-robotics && pip install "cython<3" && pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb && pip install shimmy[gym-v21] && /bin/micromamba install -c conda-forge mesalib glew glfw
RUN apt-get update && apt-get -y install patchelf
