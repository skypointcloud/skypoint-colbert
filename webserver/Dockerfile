# This is a CPU based image.

# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Poetry
RUN pip install poetry

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the working directory
COPY pyproject.toml poetry.lock* /app/

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your application's code
COPY ./webserver/ /app/

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]

