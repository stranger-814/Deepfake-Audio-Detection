# DeepFake Audio Detection
A cutting-edge deepfake audio detection web application. This project is dedicated to utilizing the Xception model, specifically trained on Mel-frequency cepstral coefficients (MFCC) features, to provide accurate and reliable detection of deepfake audio.

# Getting Started
1. Clone the repository:
   ```
   git clone https://github.com/your-username/DeepFake-Audio-Detection.git
   ```

2. Navigate to the project directory:
   ```
   cd DeepFake-Audio-Detection
   ```

3. Install Flask:
   Ensure Flask is installed on your system:
   ```
   pip install flask
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run the web application:
   ```
   python app.py
   ```

# Setup Datasets
Access the original datasets used in our project from the following URLs:
- (Bonafide) Human Voice Dataset:
  - [LJ Speech (v1.1)]( https://zenodo.org/records/5642694) - Consists of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.
  
- (Fake) Synthetic Voice Dataset:
  - [WaveFake (v1.20)]( https://keithito.com/LJ-Speech-Dataset/) - Includes 104,885 generated audio clips (16-bit PCM wav).

After downloading the datasets, extract them under `data/bonafide` and `data/fake` respectively.

# Output
Our web application generates outputs classified as "Real" or "Fake," providing a seamless experience for identifying manipulated audio content.

# Contributors
- Sandesh Lingayat : https://github.com/stranger-814 
- Sailaja Rao : https://github.com/SailajaRao
- Mugdha Kokate : https://github.com/Mugdha612
- Anish Thube : https://github.com/anisshh01
