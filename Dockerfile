# Container for building the environment
FROM condaforge/mambaforge:4.9.2-5 as conda

# Grab requirements.txt.
ADD ./hls_nrt_environment.yml /tmp/environment.yml

# Add our code
ADD . /opt/webapp/
WORKDIR /opt/webapp

RUN mamba env create --file /tmp/environment.yml && conda clean -afy

#CMD panel serve --address="0.0.0.0" --port=$PORT hls_aws_nrt.ipynb --allow-websocket-origin=range-sgx.herokuapp.com
