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


class SmartVoiceDetector:
    def __init__(self):
        """
        Initialize voice detector with TensorFlow model and smart predictions
        Compatible with both model_suara.h5 and model_suara_balanced.h5
        """
        try:
            # Try to load TensorFlow model (multiple possible names)
            model_files = ['model_suara_ultimate_fixed.h5', 'model_suara_balanced.h5']
            self.model = None

            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        self.model = tf.keras.models.load_model(model_file)
                        logger.info(f"TensorFlow model loaded from: {model_file}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_file}: {e}")

            if self.model is None:
                raise Exception("No valid TensorFlow model found!")

            # Try to load label encoder (multiple possible names)
            encoder_files = ['label_encoder_ultimate_fixed.pkl', 'label_encoder_balanced.pkl']
            self.label_encoder = None

            for encoder_file in encoder_files:
                if os.path.exists(encoder_file):
                    try:
                        with open(encoder_file, 'rb') as f:
                            self.label_encoder = pickle.load(f)
                        logger.info(f"Label encoder loaded from: {encoder_file}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {encoder_file}: {e}")

            if self.label_encoder is None:
                raise Exception("No valid label encoder found!")

            # Show available classes
            logger.info(f"Available classes: {list(self.label_encoder.classes_)}")

            # Define known vs unknown classes
            self.known_voices = [name for name in self.label_encoder.classes_
                                 if 'tidak dikenal' not in name.lower() and 'unknown' not in name.lower()]
            self.unknown_voices = [name for name in self.label_encoder.classes_
                                   if 'tidak dikenal' in name.lower() or 'unknown' in name.lower()]

            logger.info(f"Known voices: {self.known_voices}")
            logger.info(f"Unknown categories: {self.unknown_voices}")

            # Smart thresholds - Optimized for imbalanced datasets
            self.thresholds = {
                'known_voice_min': 0.3,  # Minimum confidence for known voices
                'unknown_vs_known': 0.2,  # Threshold to prefer known over unknown
                'absolute_min': 0.1,  # Absolute minimum confidence
                'known_voice_boost': 3.0,  # Boost factor for known voices
                'certainty_threshold': 0.8  # High confidence threshold
            }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_audio(self, audio_path):
        """Preprocess audio file for feature extraction"""
        try:
            # Load audio with 16kHz sample rate
            audio, sr = librosa.load(audio_path, sr=16000)

            # Normalize audio
            audio = librosa.util.normalize(audio)

            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)

            # Check minimum length (0.5 seconds)
            if len(audio) < sr * 0.5:
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

            # Validate feature length
            if len(combined) != 80:
                logger.error(f"Expected 80 features, got {len(combined)}")
                return None

            return combined
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def smart_predict(self, features):
        """Smart prediction with bias correction towards known voices"""
        if features is None:
            return {"error": "Couldn't process audio file", "confidence": 0.0}

        try:
            # Reshape features for model input
            features_reshaped = features.reshape(1, -1)

            # Make prediction
            raw_predictions = self.model.predict(features_reshaped, verbose=0)[0]

            # Create probability dictionary
            all_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                all_probabilities[class_name] = float(raw_predictions[i])

            # Apply smart logic to reduce unknown bias
            adjusted_predictions = self._apply_smart_logic(raw_predictions, all_probabilities)

            # Get final prediction
            predicted_class_idx = np.argmax(adjusted_predictions)
            confidence = float(np.max(adjusted_predictions))
            predicted_class = self.label_encoder.classes_[predicted_class_idx]

            # Create detailed result
            result = {
                "prediction": predicted_class,
                "confidence": confidence,
                "confidence_percent": confidence * 100,
                "raw_probabilities": all_probabilities,
                "adjusted_probabilities": {
                    name: float(adjusted_predictions[i])
                    for i, name in enumerate(self.label_encoder.classes_)
                },
                "prediction_logic": self._explain_prediction(predicted_class, confidence, all_probabilities),
                "certainty_level": self._get_certainty_level(confidence)
            }

            # Final confidence check
            if confidence < self.thresholds['absolute_min']:
                result["prediction"] = "uncertain"
                result["warning"] = "Very low confidence prediction"

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": "Prediction failed", "confidence": 0.0}

    def _apply_smart_logic(self, raw_predictions, all_probabilities):
        """Apply smart logic to reduce unknown voice bias"""
        adjusted = raw_predictions.copy()

        # Get best known voice and best unknown voice
        known_probs = {name: prob for name, prob in all_probabilities.items()
                       if name in self.known_voices}
        unknown_probs = {name: prob for name, prob in all_probabilities.items()
                         if name in self.unknown_voices}

        if known_probs and unknown_probs:
            best_known = max(known_probs.items(), key=lambda x: x[1])
            best_unknown = max(unknown_probs.items(), key=lambda x: x[1])

            logger.debug(f"Best known: {best_known[0]} ({best_known[1]:.3f})")
            logger.debug(f"Best unknown: {best_unknown[0]} ({best_unknown[1]:.3f})")

            # If unknown is winning but known is reasonably close, boost known
            if (best_unknown[1] > best_known[1] and
                    best_known[1] > self.thresholds['unknown_vs_known']):
                # Find index of best known voice
                best_known_idx = list(self.label_encoder.classes_).index(best_known[0])

                # Apply boost to known voice
                boost_factor = self.thresholds['known_voice_boost']
                adjusted[best_known_idx] *= boost_factor

                # Re-normalize probabilities
                adjusted = adjusted / np.sum(adjusted)

                logger.info(
                    f"Applied boost to {best_known[0]}: {raw_predictions[best_known_idx]:.3f} → {adjusted[best_known_idx]:.3f}")

        return adjusted

    def _explain_prediction(self, predicted_class, confidence, all_probabilities):
        """Explain the prediction logic"""
        if predicted_class in self.known_voices:
            if confidence > self.thresholds['certainty_threshold']:
                return f"High confidence match for known voice: {predicted_class}"
            elif confidence > self.thresholds['known_voice_min']:
                return f"Moderate confidence for known voice: {predicted_class}"
            else:
                return f"Low confidence for known voice: {predicted_class}"
        elif predicted_class in self.unknown_voices:
            # Check if any known voice was close
            known_probs = {name: prob for name, prob in all_probabilities.items()
                           if name in self.known_voices}
            if known_probs:
                best_known = max(known_probs.items(), key=lambda x: x[1])
                if best_known[1] > 0.3:
                    return f"Predicted unknown, but {best_known[0]} had {best_known[1]:.1%} confidence"
            return f"Predicted as unknown voice category"
        else:
            return "Uncertain prediction"

    def _get_certainty_level(self, confidence):
        """Get certainty level description"""
        if confidence > self.thresholds['certainty_threshold']:
            return "HIGH"
        elif confidence > self.thresholds['known_voice_min']:
            return "MEDIUM"
        elif confidence > self.thresholds['absolute_min']:
            return "LOW"
        else:
            return "VERY_LOW"

    def predict(self, features):
        """Main predict method (for compatibility)"""
        return self.smart_predict(features)

    def update_thresholds(self, new_thresholds):
        """Update prediction thresholds"""
        for key, value in new_thresholds.items():
            if key in self.thresholds:
                self.thresholds[key] = float(value)
                logger.info(f"Updated threshold {key} to {value}")


# Flask app setup
app = Flask(__name__)

# Initialize voice detector
try:
    detector = SmartVoiceDetector()
    logger.info("Smart voice detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize voice detector: {e}")
    detector = None

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/api/detect/suara', methods=['POST'])
def predict_voice():
    """Predict voice from uploaded audio file with smart logic"""
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
        result = detector.smart_predict(features)

        # Clean up temporary file
        if os.path.exists(original_path):
            os.remove(original_path)

        # Prepare response
        response = {
            'status': 'success',
            'username_requested': username,
            'model_type': 'tensorflow_smart',
            **result
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get information about the model and available classes"""
    if not detector:
        return jsonify({'error': 'Voice detector not initialized'}), 500

    try:
        return jsonify({
            'available_classes': list(detector.label_encoder.classes_),
            'known_voices': detector.known_voices,
            'unknown_categories': detector.unknown_voices,
            'num_classes': len(detector.label_encoder.classes_),
            'model_type': 'tensorflow_smart',
            'feature_dimensions': 80,
            'thresholds': detector.thresholds,
            'model_info': {
                'input_shape': detector.model.input.shape[1:],
                'output_shape': detector.model.output.shape[1:],
                'total_parameters': detector.model.count_params()
            }
        })
    except Exception as e:
        return jsonify({'error': f'Unable to get info: {str(e)}'}), 500


@app.route('/api/thresholds/adjust', methods=['POST'])
def adjust_thresholds():
    """Adjust prediction thresholds on the fly"""
    if not detector:
        return jsonify({'error': 'Voice detector not initialized'}), 500

    try:
        data = request.get_json()
        detector.update_thresholds(data)

        return jsonify({
            'status': 'success',
            'updated_thresholds': detector.thresholds,
            'message': 'Thresholds updated successfully'
        })

    except Exception as e:
        return jsonify({'error': f'Threshold update error: {str(e)}'}), 500


@app.route('/api/test/batch', methods=['POST'])
def batch_test():
    """Test multiple audio files in a batch"""
    if not detector:
        return jsonify({'error': 'Voice detector not initialized'}), 500

    try:
        files = request.files.getlist('audio_files')
        if not files:
            return jsonify({'error': 'No audio files provided'}), 400

        results = []
        for i, file in enumerate(files):
            # Save file temporarily
            temp_path = os.path.join(UPLOAD_FOLDER, f"batch_{i}_{file.filename}")
            file.save(temp_path)

            try:
                # Convert if needed
                if not temp_path.lower().endswith('.wav'):
                    audio = AudioSegment.from_file(temp_path)
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    wav_path = temp_path.rsplit('.', 1)[0] + '.wav'
                    audio.export(wav_path, format='wav')
                    os.remove(temp_path)
                    temp_path = wav_path

                # Predict
                features = detector.extract_features(temp_path)
                result = detector.smart_predict(features)

                results.append({
                    'filename': file.filename,
                    'prediction': result.get('prediction', 'error'),
                    'confidence': result.get('confidence', 0.0),
                    'certainty_level': result.get('certainty_level', 'UNKNOWN')
                })

            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return jsonify({
            'status': 'success',
            'results': results,
            'total_files': len(files)
        })

    except Exception as e:
        return jsonify({'error': f'Batch test error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None and detector.model is not None,
        'model_type': 'tensorflow_smart',
        'version': '2.0'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    if detector:
        print("🎤 Smart Voice Recognition Server Starting...")
        print(f"🧠 Model Type: TensorFlow Neural Network")
        print(f"👥 Known Voices: {detector.known_voices}")
        print(f"❓ Unknown Categories: {detector.unknown_voices}")
        print(f"⚙️  Smart Thresholds: {detector.thresholds}")
        print("🌐 Server running on http://localhost:15018")
        print("\n📋 Available endpoints:")
        print("  POST /api/detect/suara      - Upload audio for voice recognition")
        print("  GET  /api/model/info        - Get model information")
        print("  POST /api/thresholds/adjust - Adjust prediction thresholds")
        print("  POST /api/test/batch        - Batch test multiple files")
        print("  GET  /health               - Health check")

        app.run(debug=True, host="0.0.0.0", port=15018)
    else:
        print("❌ Cannot start server - voice detector initialization failed")
        print("Make sure these files exist:")
        print("  - model_suara_ultimate_fixed.h5 or model_suara_balanced.h5")
        print("  - label_encoder_ultimate_fixed.pkl or label_encoder_balanced.pkl")