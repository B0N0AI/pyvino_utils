FROM mmphego/intel-openvino
WORKDIR /app
COPY . /app
RUN pip install .[dev]
