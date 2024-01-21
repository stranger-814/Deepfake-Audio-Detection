# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Xception model
model_path = 'C:/Users/Mayuresh/Desktop/2112024/xception_model.h5'
loaded_model = load_model(model_path)


# Function to extract MFCC features and save as heatmap
def extract_mfcc(file_path, output_directory):
    # Your implementation for extracting MFCC features goes here
    # This is a basic example, make sure to customize it based on your needs
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Plotting the MFCC as a heatmap
    plt.figure(figsize=(8, 6))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC Heatmap')

    # Check if the output directory exists, if not, create it
    os.makedirs(output_directory, exist_ok=True)

    # Save the MFCC heatmap
    output_path = os.path.join(output_directory, os.path.basename(file_path) + '_mfcc.png')
    plt.savefig(output_path)
    plt.close()

    return mfcc


# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


# Define the route for audio file upload and processing
@app.route('/detect', methods=['POST'])
def detect_audio():
    # Check if the 'audio' file is in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    # Save the uploaded audio file
    audio_file = request.files['audio']
    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)

    # Extract MFCC features and save as a heatmap
    mfcc_features = extract_mfcc(audio_path, output_directory='./uploads/mfcc')

    # Load the MFCC heatmap image
    mfcc_image_path = os.path.join('uploads', 'mfcc', audio_file.filename + '_mfcc.png')
    img = image.load_img(mfcc_image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction using the loaded Xception model
    prediction = loaded_model.predict(img_array)
    result = "The Audio is Fake" if prediction[0][0] > 0.5 else "The Audio is Bonafide"

    # Return the result as JSON
    return jsonify({'result': result})


if __name__ == '__main__':
    # Run the app on localhost:5000
    app.run(debug=True)
