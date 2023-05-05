FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-9
WORKDIR /root
RUN pip install --no-cache-dir chess numpy
COPY . .
ENTRYPOINT ["python", "train_self_play.py"]
