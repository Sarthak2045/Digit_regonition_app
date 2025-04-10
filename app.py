from flask import Flask, render_template, request, jsonify
import torch
import base64
import io
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os

# Import your model definition
class DigitRecognitionCNN(torch.nn.Module):
    def __init__(self):
        super(DigitRecognitionCNN, self).__init__()
        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        
        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        
        # Third convolutional layer
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3)
        
        # Calculate input size for first fully connected layer
        self.fc1 = torch.nn.Linear(64 * 3 * 3, 128)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        # First convolutional block
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Second convolutional block
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Third convolutional layer
        x = self.relu(self.conv3(x))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 3 * 3)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

app = Flask(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = None

def load_model():
    global model
    model = DigitRecognitionCNN().to(device)
    model_path = 'mnist_digit_recognition_model.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    else:
        print("Model file not found!")
    
    return model

# Preprocess image function
def preprocess_image(image):
    # Convert image to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    
    # Apply transformations for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Apply transformations
    img_tensor = transform(image)
    return img_tensor

# Predict function
def predict(img_tensor):
    with torch.no_grad():
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get the predicted class
        _, predicted = torch.max(output, 1)
        predicted_digit = predicted.item()
        confidence = probabilities[predicted_digit].item() * 100
        
        # Get top 3 predictions
        top3_values, top3_indices = torch.topk(probabilities, 3)
        top3_predictions = [(idx.item(), val.item() * 100) for idx, val in zip(top3_indices, top3_values)]
    
    return {
        'digit': predicted_digit,
        'confidence': confidence,
        'top3': top3_predictions
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Open the image
        img = Image.open(file.stream)
        
        # Preprocess image
        img_tensor = preprocess_image(img)
        
        # Make prediction
        result = predict(img_tensor)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model()
    app.run(debug=True)