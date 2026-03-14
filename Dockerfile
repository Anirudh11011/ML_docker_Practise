FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (including train.py and app.py)
COPY . .

# RUN THE TRAINING during the build process
# This creates mnist_model.pth INSIDE the image layers
RUN python train.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]