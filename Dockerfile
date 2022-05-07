FROM usgs/miniconda3

# Grab requirements.txt.
ADD ./hls_nrt_environment.yml /tmp/environment.yml

# Add our code
ADD . /opt/webapp/
WORKDIR /opt/webapp

RUN conda env update --file /tmp/environment.yml --prune

CMD panel serve --address="0.0.0.0" --port=$PORT hls_aws_nrt.ipynb --allow-websocket-origin=range-sgx.herokuapp.com
