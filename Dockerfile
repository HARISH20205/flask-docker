FROM python:3.10-slim
COPY  . /app
EXPOSE 5000
WORKDIR /app
RUN pip install -r requirements.txt
CMD app.py