# Base image: includes Ollama preinstalled
FROM ollama/ollama:latest

# Install Python + pip
RUN apt update && apt install -y python3 python3-pip git

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies (override PEP 668 restriction)
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Pull your local model (optional, can skip for faster build)
RUN ollama pull llama3.2

# Expose Streamlit and Ollama ports
EXPOSE 8501
EXPOSE 11434

# Start Ollama (background) + Streamlit (foreground)
CMD ollama serve & \
    streamlit run app3.py --server.port 8501 --server.address 0.0.0.0
