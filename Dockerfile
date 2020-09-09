FROM mmphego/intel-openvino
WORKDIR /app
COPY . /app
RUN pip install -r requirements-dev.txt
RUN pip install .
