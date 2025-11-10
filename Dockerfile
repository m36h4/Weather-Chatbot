# Base image: includes Ollama preinstalled
FROM ollama/ollama:latest

# Install Python + dependencies
RUN apt update && apt install -y python3 python3-pip git

# Set working directory
WORKDIR /app

# Copy your app into the container
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8501
EXPOSE 11434

# Pull your Ollama model at build time (optional)
RUN ollama pull llama3.2

# Start both Ollama and Streamlit
CMD ollama serve & \
    streamlit run app3.py --server.port 8501 --server.address 0.0.0.0
