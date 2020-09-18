FROM mmphego/intel-openvino
WORKDIR /app
COPY . /app
USER root
RUN pip install .[dev]
USER vino
