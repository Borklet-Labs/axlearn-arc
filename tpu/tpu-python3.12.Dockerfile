# syntax=docker/dockerfile:1

ARG TARGET=base
ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS base

# Install curl and gpupg first so that we can use them to install google-cloud-cli.
# Any RUN apt-get install step needs to have apt-get update otherwise stale package
# list may occur when previous apt-get update step is cached. See here for more info:
# https://docs.docker.com/build/building/best-practices/#apt-get
RUN apt-get update && apt-get upgrade -y && apt-get install -y curl gnupg && apt clean -y

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates gcc g++ \
      git screen ca-certificates google-perftools google-cloud-cli python3.12-venv && apt clean -y

# Setup.
RUN mkdir -p /var/arc
WORKDIR /var/arc
# Setup venv to suppress pip warnings.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install dependencies.
RUN pip install --upgrade pip && pip install uv flit && pip cache purge
# Ensure we are pulling the custom wheels required for Python 3.12
ENV UV_FIND_LINKS="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html,https://storage.googleapis.com/axlearn-wheels/wheels.html"
# Copy the test setup to the image
COPY . .

# Make the startup script executable.
RUN chmod +x /var/arc/tpu-startup.sh

# Set the final working directory.
WORKDIR /root

# Set the default command to run the script when the container starts.
# We use the full path because the script was copied to /var/arc.
CMD ["/var/arc/tpu-jobset-multislice-startup.sh"]
