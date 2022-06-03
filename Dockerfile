
#FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         ca-certificates \
         libboost-all-dev \
         libjpeg-dev \
         libpng-dev \
         #validators \
         python3-setuptools \
         libgl1-mesa-glx libsm6 libxrender1 libxext-dev \
         nginx libgl1 && \
     rm -rf /var/lib/apt/lists/*

RUN conda install tornado=5.0.2 ply=3.11 
RUN pip install tqdm gdown scikit-learn==0.22 scipy lpips dlib opencv-python pandas matplotlib

RUN cd /home && \
    git clone https://github.com/hqqxyy/face_attr_ensemble.git app

COPY models/ /home/app/models/
#RUN mkdir -p /home/.cache/torch/hub/checkpoints/ && \
#   mv /home/app/models/vgg16-397923af.pth /home/.cache/torch/hub/checkpoints/

COPY *.patch /home/app/


RUN bash -c 'mkdir -p /home/app/input/face/{ref,source}'
RUN cd /home/app && \
    git config --global user.email "you@example.com" && \
    git config --global user.name "Your Name" && \
    git am *.patch && \
    rm -f *.patch

#COPY --from=hair3d_common /home/*.py /home/app/
COPY *.py /home/app/
COPY *.conf /etc/nginx/ 

RUN rm -fr /usr/bin/python*

CMD  ["/opt/conda/bin/python","/home/app/webs.py"]
WORKDIR /home/app

####
ARG  USER=docker
ARG  GROUP=docker
ARG  UID
ARG  GID
## must use ; here to ignore user exist status code
RUN  [ ${GID} -gt 0 ] && groupadd -f -g ${GID} ${GROUP}; \
     [ ${UID} -gt 0 ] && useradd -d /home -M -g ${GID} -K UID_MAX=${UID} -K UID_MIN=${UID} ${USER}; \
     chown -R ${UID}:${GID} /home && \
     touch /var/run/nginx.pid && \
     mkdir -p /var/log/nginx /var/lib/nginx && \
     chown ${UID}:${GID} $(find /home -maxdepth 2 -type d -print) /var/run/nginx.pid && \
     chown -R ${UID}:${GID} /var/log/nginx /var/lib/nginx
USER ${UID}
####

