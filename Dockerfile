# pull and prepare the container
#FROM python:3.10-buster
FROM ubuntu:20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# install Ubuntu dependency
RUN apt-get update && apt-get install -y gnupg
RUN apt-get install -y curl

# install git
RUN apt-get install -y git

# Install system dependencies
RUN set -e; \
    apt-get update -y && apt-get install -y \
    tini \
    lsb-release; \
    gcsFuseRepo=gcsfuse-`lsb_release -c -s`; \
    echo "deb https://packages.cloud.google.com/apt $gcsFuseRepo main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key add -; \
    apt-get update; \
    apt-get install -y gcsfuse \
    && apt-get clean
	
# Set fallback mount directory
ENV MNT_DIR /mnt/gcs
ENV BUCKET hls_nrt

# install production dependencies
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
	
# Grab production requirements
ADD ./hls_nrt_environment.yml /tmp/environment.yml

# Add our code and set working directory
#ADD . /opt/webapp/
#WORKDIR /opt/webapp

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# create production environment
RUN conda env create -p /env --file /tmp/environment.yml && \
  conda clean -afy && \
  conda init

# clean up any Windows-style paths 
# see: https://stackoverflow.com/questions/29045140/env-bash-r-no-such-file-or-directory/29045187#29045187
# Installs dos2unix Linux
RUN apt-get install -y dos2unix 
# recursively removes windows related stuff
RUN find . -type f -exec dos2unix {} \; 

# Ensure the script is executable
RUN chmod +x /app/gcsfuse_run.sh

# Use tini to manage zombie processes and signal forwarding
# https://github.com/krallin/tini
ENTRYPOINT ["/usr/bin/tini", "--"] 

# Pass the startup script as arguments to Tini
CMD ["/app/gcsfuse_run.sh"]
