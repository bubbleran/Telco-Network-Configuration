# Base image with Python 3.10
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

# Install dependencies for docker CLI
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
apt-get update && \
apt-get install -y docker-ce-cli

# Copy requirements and install them
COPY requirements.txt .
COPY ./5g-sa-nr-sim ./5g-sa-nr-sim

RUN pip install --no-cache-dir -r requirements.txt

ARG BUBBLERAN_HOST_PWD
ENV BUBBLERAN_HOST_PWD=$BUBBLERAN_HOST_PWD
RUN mkdir -p "$BUBBLERAN_HOST_PWD" && cp -r ./5g-sa-nr-sim "$BUBBLERAN_HOST_PWD/5g-sa-nr-sim"

WORKDIR /workspace

CMD ["streamlit", "run", "telco_planner_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
