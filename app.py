import os
import pickle
import numpy as np
import librosa
import warnings
import logging
from flask import Flask, request, jsonify
from pydub import AudioSegment
import tensorflow as tf

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceDetector:
    def __init__(self, model_path='model_suara.h5', label_encoder_path='label_encoder.pkl'):
        """
        Initialize voice detector with your trained TensorFlow model
        """
        try:
            # Load TensorFlow model
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"TensorFlow model loaded from: {model_path}")

            # Load label encoder
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Label encoder loaded from: {label_encoder_path}")

            # Show available classes
            logger.info(f"Available classes: {list(self.label_encoder.classes_)}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_audio(self, audio_path):
        """Preprocess audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            audio = librosa.util.normalize(audio)
            audio, _ = librosa.effects.trim(audio, top_db=20)

            if len(audio) < sr * 0.5:  # Less than 0.5 seconds
                logger.warning(f"Audio too short: {len(audio) / sr:.2f} seconds")
                return None, sr

            return audio, sr
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None, None

    def extract_features(self, audio_path):
        """Extract MFCC features from audio file (same as training)"""
        try:
            audio, sr = self.preprocess_audio(audio_path)
            if audio is None:
                return None

            # Extract MFCC features (same parameters as training)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

            # Calculate mean and standard deviation
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.std(mfccs.T, axis=0)

            # Combine features
            combined = np.hstack([mfccs_mean, mfccs_std])

            if len(combined) != 80:
                logger.error(f"Expected 80 features, got {len(combined)}")
                return None

            return combined
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def predict(self, features):
        """Make prediction on extracted features"""
        if features is None:
            return {"error": "Couldn't process audio file", "confidence": 0.0}

        try:
            # Reshape features for model input
            features_reshaped = features.reshape(1, -1)

            # Make prediction
            predictions = self.model.predict(features_reshaped, verbose=0)

            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            # Convert back to class name
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]

            # Create detailed result
            result = {
                "prediction": predicted_class,
                "confidence": confidence,
                "confidence_percent": confidence * 100,
                "all_probabilities": {}
            }

            # Add all class probabilities
            for i, class_name in enumerate(self.label_encoder.classes_):
                result["all_probabilities"][class_name] = float(predictions[0][i])

            # Determine if prediction is reliable
            if confidence < 0.6:
                result["prediction"] = "unknown"
                result["warning"] = "Low confidence prediction"

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": "Prediction failed", "confidence": 0.0}


# Flask app setup
app = Flask(__name__)

# Initialize detector
try:
    detector = VoiceDetector()
    logger.info("Voice detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize voice detector: {e}")
    detector = None

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/api/detect/suara', methods=['POST'])
def predict_voice():
    """Predict voice from uploaded audio file"""
    if not detector:
        return jsonify({'error': 'Voice detector not initialized'}), 500

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        username = request.form.get('username', 'unknown')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        original_path = os.path.join(UPLOAD_FOLDER, f"{username}_{file.filename}")
        file.save(original_path)

        try:
            # Convert to WAV format if needed
            if not original_path.lower().endswith('.wav'):
                audio = AudioSegment.from_file(original_path)
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                wav_path = original_path.rsplit('.', 1)[0] + '.wav'
                audio.export(wav_path, format='wav')
                os.remove(original_path)
                original_path = wav_path
        except Exception as e:
            if os.path.exists(original_path):
                os.remove(original_path)
            return jsonify({'error': f'Audio conversion error: {str(e)}'}), 400

        # Extract features and predict
        features = detector.extract_features(original_path)
        result = detector.predict(features)

        # Clean up temporary file
        if os.path.exists(original_path):
            os.remove(original_path)

        # Prepare response
        response = {
            'status': 'success',
            'username_requested': username,
            'model_type': 'tensorflow',
            **result
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/api/info', methods=['GET'])
def get_model_info():
    """Get information about the model and available classes"""
    if not detector:
        return jsonify({'error': 'Voice detector not initialized'}), 500

    try:
        return jsonify({
            'available_classes': list(detector.label_encoder.classes_),
            'num_classes': len(detector.label_encoder.classes_),
            'model_type': 'tensorflow',
            'model_accuracy': '97.94%',  # From your training results
            'feature_dimensions': 80,
            'total_training_samples': 1697  # From your training results
        })
    except Exception as e:
        return jsonify({'error': f'Unable to get info: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None and detector.model is not None,
        'model_type': 'tensorflow'
    })


@app.route('/api/test', methods=['POST'])
def test_prediction():
    """Test endpoint for debugging"""
    if not detector:
        return jsonify({'error': 'Voice detector not initialized'}), 500

    try:
        # This endpoint can be used to test the model with known audio files
        test_file_path = request.json.get('file_path')

        if not test_file_path or not os.path.exists(test_file_path):
            return jsonify({'error': 'Test file not found'}), 400

        features = detector.extract_features(test_file_path)
        result = detector.predict(features)

        return jsonify({
            'status': 'test_success',
            'test_file': test_file_path,
            **result
        })

    except Exception as e:
        return jsonify({'error': f'Test error: {str(e)}'}), 500


if __name__ == '__main__':
    if detector:
        print("🎤 Voice Recognition Server Starting...")
        print(f"🧠 Model Classes: {list(detector.label_encoder.classes_)}")
        print("🌐 Server running on http://localhost:15018")
        print("\n📋 Available endpoints:")
        print("  POST /api/detect/suara - Upload audio for voice recognition")
        print("  GET  /api/info         - Get model information")
        print("  GET  /health          - Health check")
        print("  POST /api/test        - Test with local file")

        app.run(debug=True, host="0.0.0.0", port=15018)
    else:
        print("❌ Cannot start server - voice detector initialization failed")
        print("Make sure these files are present:")
        print("  - model_suara.h5")
        print("  - label_encoder.pkl")