FROM nvidia/cuda:12.3.1-devel-ubuntu20.04
# the base image is missing some C library dependencies
# I can identify those dependencies and add them to the base image
# the current devel image is too large

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive


# Install software-properties-common to add PPAs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    libglib2.0-0

# Add deadsnakes PPA for newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3.9-dev python3-pip

# Update alternatives to use Python 3.9 as the default python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1


# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Poetry
# RUN pip install poetry

# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the working directory
# COPY pyproject.toml poetry.lock* /app/

# Copy the rest of your application's code
COPY ./webserver/ /app/

# Install project dependencies
# RUN poetry config virtualenvs.create false \
#    && poetry install --no-interaction --no-root --without dev
RUN pip install uvicorn fastapi torch
RUN pip install pyarrow==14.0.0
RUN pip install colbert-ai
RUN pip install ragstack-ai

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]
