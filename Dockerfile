FROM python:3.8-slim
RUN mkdir /app && chmod 777 /app
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]