# Container for building the environment
FROM condaforge/mambaforge:4.9.2-5 as conda

# update
RUN apt-get update

# get dependencies that don't seem to be installing via mamba
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Grab requirements.txt.
ADD ./hls_nrt_environment.yml /tmp/environment.yml

# Add our code and set working directory
ADD . /opt/webapp/
WORKDIR /opt/webapp

RUN conda env create -p /env --file /tmp/environment.yml && \
  conda clean -afy && \
  conda init

# added after getting error during rasterio import
#RUN export LD_PRELOAD=/env/lib/python3.10/site-packages/rasterio/../../../libgdal.so.30
#RUN unset LD_LIBRARY_PATH

#EXPOSE 8080

#CMD ["/env/bin/panel", "serve", "hls_gcloud_app.ipynb", "--port", "8080", "--num-procs", "4", "--allow-websocket-origin", "*"]

CMD ["/env/bin/panel", "serve", "hls_gcloud_app.ipynb", "--address", "0.0.0.0", "--port", "8080", "--allow-websocket-origin", "*"]
