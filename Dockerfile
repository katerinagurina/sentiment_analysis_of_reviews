FROM python:3.8.11
WORKDIR /workdir/

COPY . .

USER root

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -U -e .

# Run the application
CMD ["python3", "-m", "uvicorn", "app.app:app","--host", "0.0.0.0", "--port", "80"]

