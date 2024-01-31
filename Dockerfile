FROM python:3.10

ENV PYTHONBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

# Dependency for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY ./requirements.txt /opt/program/requirements.txt

# Installing pytorch separately because rmn installs GPU version, leading to redundant extra 3+ GB image size
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# source code
COPY ./src /opt/program/
COPY ./src/models   /opt/program/src/models

# RMN model requirements
COPY ./pretrained_ckpt /opt/program/
COPY ./deploy.prototxt.txt /opt/program/
COPY ./res10_300x300_ssd_iter_140000.caffemodel /opt/program/

ENTRYPOINT [ "gunicorn", "-b", ":5000", "serve:app" ]