FROM gcr.io/deeplearning-platform-release/tf-cpu.2-8
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer
COPY requirements.txt ./

# Install production dependencies.
RUN pip install -r requirements.txt


# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]
