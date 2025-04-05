# First stage (without GPU)
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
WORKDIR /root
RUN apt-get update && \
    apt-get install -y software-properties-common openssl curl wget git gnupg2 build-essential zlib1g-dev \
    libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.13 python3.13-venv python3.13-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Generate a 384-bit ECC private key and self-signed certificate (valid for 25 years)
COPY ./openssl.conf /etc/ssl/openssl.conf
RUN openssl ecparam -name secp384r1 -genkey -out /etc/ssl/private/server.key && \
    openssl req -new -x509 -key /etc/ssl/private/server.key -out /etc/ssl/certs/server.crt -days 9125 \
    -subj "/CN=localhost" -extensions v3_req -config /etc/ssl/openssl.cnf

# Create python environment
RUN /usr/bin/python3.13 -m venv /root/environment
RUN /root/environment/bin/python3 -m pip install --upgrade pip

# Install python dependancies
COPY ./requirements.txt /root/requirements.txt
RUN /root/environment/bin/python3 -m pip install -r /root/requirements.txt

# Download models
COPY ./download.py /root/download.py
RUN /root/environment/bin/python3 /root/download.py

# Clone extra python dependancies
RUN git clone https://github.com/nomic-ai/megablocks.git megablocks

# Copy final files
COPY ./setup.sh /root/setup.sh
COPY ./server.py /root/server.py
COPY ./ai.py /root/ai.py
COPY ./download.py /root/download.py

# Default command
WORKDIR /root
ENTRYPOINT []
CMD ["/root/environment/bin/python3", "/root/server.py"]
