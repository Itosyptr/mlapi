import os
import pickle
import numpy as np
import librosa
import warnings
import logging
from flask import Flask, request, jsonify
from pydub import AudioSegment

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(_name_)


class VoiceDetector:

    def _init_(self, model_path='model_knn_suara.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            logger.info("Model loaded successfully")

    def preprocess_audio(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            audio = librosa.util.normalize(audio)
            audio, _ = librosa.effects.trim(audio, top_db=20)
            if len(audio) < sr:
                logger.warning(f"Audio too short: {len(audio)/sr:.2f} seconds")
                return None, sr
            return audio, sr
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None, None

    def extract_features(self, audio_path):
        try:
            audio, sr = self.preprocess_audio(audio_path)
            if audio is None:
                return None
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            combined = np.hstack([mfccs_mean, mfccs_std])
            if len(combined) != 80:
                logger.error(f"Expected 80 features, got {len(combined)}")
                return None
            return combined
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def predict(self, features):
        if features is None:
            return "Error: Couldn't process audio file"
        try:
            pred = self.model.predict(features.reshape(1, -1))[0]
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features.reshape(1, -1))[0]
                if np.max(proba) < 0.6:
                    return "unknown"
            return pred
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error: Prediction failed"


# Flask app
app = Flask(_name_)
MODEL_PATH = 'model_knn_suara.pkl'
detector = VoiceDetector(MODEL_PATH)
UPLOAD_FOLDER = 'temp_uploads'
TRAIN_FOLDER = 'train_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)


@app.route('/api/detect/suara', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        username = request.form.get('username', 'unknown')
        original_path = os.path.join(UPLOAD_FOLDER, f"{username}_{file.filename}")
        file.save(original_path)

        try:
            audio = AudioSegment.from_file(original_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            wav_path = original_path.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format='wav')
        except Exception as e:
            os.remove(original_path)
            return jsonify({'error': f'Conversion error: {str(e)}'}), 400

        features = detector.extract_features(wav_path)
        os.remove(original_path)
        os.remove(wav_path)

        if features is None:
            return jsonify({'error': 'Audio processing failed'}), 400

        prediction = detector.predict(features)
        return jsonify({'prediction': prediction, 'username_requested': username, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/api/train/suara', methods=['POST'])
def train_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        username = request.form.get('username')

        if not username:
            return jsonify({'error': 'Username is required'}), 400

        user_dir = os.path.join(TRAIN_FOLDER, username)
        os.makedirs(user_dir, exist_ok=True)

        original_path = os.path.join(user_dir, file.filename)
        file.save(original_path)

        try:
            audio = AudioSegment.from_file(original_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            base_name = os.path.splitext(file.filename)[0]
            wav_path = os.path.join(user_dir, base_name + '.wav')
            audio.export(wav_path, format='wav')
            os.remove(original_path)
        except Exception as e:
            return jsonify({'error': f'Conversion error: {str(e)}'}), 400

        return jsonify({'status': 'success', 'message': f'Training audio saved for user {username}'})
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': detector.model is not None})


if _name_ == '_main_':
    app.run(debug=True, host="0.0.0.0", port=15018)