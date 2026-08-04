FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dcl_core.py mcp_server.py ./

# Smithery выставит PORT=8081 при запуске контейнера
ENV PORT=8081
EXPOSE 8081

CMD ["python", "mcp_server.py"]
