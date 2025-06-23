#!/usr/bin/env python3
"""
VAD Voice Recognition System - Testing & Deployment Script
Advanced testing and deployment utilities for the VAD system
"""

import os
import sys
import requests
import json
import time
import glob
import argparse
from pathlib import Path


# ========================================
# API TESTING FUNCTIONS
# ========================================

class VADSystemTester:
    """Test the VAD Voice Recognition API"""

    def __init__(self, base_url="http://localhost:15018"):
        self.base_url = base_url
        self.session = requests.Session()

    def test_health(self):
        """Test system health"""
        print("🏥 Testing System Health...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("  ✅ System is healthy")
                print(f"  📊 Models loaded: {data.get('total_models', 0)}")
                print(f"  🧠 VAD ensemble: {'✅' if data.get('vad_ensemble_loaded') else '❌'}")
                return True
            else:
                print(f"  ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Connection error: {e}")
            return False

    def test_system_info(self):
        """Test system information endpoint"""
        print("\n📋 Getting System Information...")
        try:
            response = self.session.get(f"{self.base_url}/api/system/info")
            if response.status_code == 200:
                data = response.json()
                print("  ✅ System info retrieved")
                print(f"  🎯 System: {data.get('system')}")
                print(f"  📦 Version: {data.get('version')}")
                print(f"  🧠 Models: {', '.join(data.get('models_available', []))}")
                print(f"  🎵 Formats: {', '.join(data.get('supported_formats', []))}")
                return True
            else:
                print(f"  ❌ System info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def test_voice_detection(self, audio_file, username="test_user"):
        """Test voice detection with audio file"""
        print(f"\n🎤 Testing Voice Detection: {os.path.basename(audio_file)}")

        if not os.path.exists(audio_file):
            print(f"  ❌ Audio file not found: {audio_file}")
            return False

        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                data = {'username': username}

                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/api/detect/suara",
                    files=files, data=data
                )
                processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Detection successful ({processing_time:.2f}s)")
                print(f"  🎯 Prediction: {result.get('prediction')}")
                print(f"  💯 Confidence: {result.get('confidence_percent', 0):.1f}%")
                print(f"  🔍 VAD Status: {'✅ Passed' if result.get('vad_passed') else '❌ Rejected'}")

                if result.get('prediction') == 'REJECTED':
                    print(f"  ⚠️  Rejection reason: {result.get('rejection_reason')}")

                if 'model_details' in result:
                    print("  📊 Individual model predictions:")
                    for model, pred in result['model_details'].items():
                        print(f"    {model}: {pred['class']} ({pred['confidence_percent']:.1f}%)")

                return True
            else:
                print(f"  ❌ Detection failed: {response.status_code}")
                print(f"  📝 Response: {response.text}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def test_audio_analysis(self, audio_file, username="test_user"):
        """Test detailed audio analysis"""
        print(f"\n🔬 Testing Audio Analysis: {os.path.basename(audio_file)}")

        if not os.path.exists(audio_file):
            print(f"  ❌ Audio file not found: {audio_file}")
            return False

        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                data = {'username': username}

                response = self.session.post(
                    f"{self.base_url}/api/analyze/audio",
                    files=files, data=data
                )

            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Analysis successful")
                print(f"  ⏱️  Duration: {result.get('duration_seconds', 0):.2f}s")
                print(f"  🎵 Classification: {result.get('classification')}")
                print(f"  💯 Confidence: {result.get('confidence', 0):.2f}")
                print(f"  🗣️  Is Voice: {'✅' if result.get('is_voice') else '❌'}")

                if 'detected_issues' in result:
                    print(f"  ⚠️  Issues detected: {', '.join(result['detected_issues'])}")

                # Show key metrics
                details = result.get('details', {})
                if 'energy' in details:
                    energy = details['energy']
                    print(f"  🔊 Energy: {energy.get('rms_energy_db', 0):.1f} dB")

                if 'pitch' in details:
                    pitch = details['pitch']
                    print(f"  🎵 Pitch confidence: {pitch.get('pitch_confidence', 0):.2f}")
                    print(f"  🗣️  Voiced ratio: {pitch.get('voiced_ratio', 0):.2f}")

                return True
            else:
                print(f"  ❌ Analysis failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def test_training_upload(self, audio_file, username="test_user"):
        """Test training audio upload"""
        print(f"\n📚 Testing Training Upload: {os.path.basename(audio_file)}")

        if not os.path.exists(audio_file):
            print(f"  ❌ Audio file not found: {audio_file}")
            return False

        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                data = {'username': username}

                response = self.session.post(
                    f"{self.base_url}/api/train/suara",
                    files=files, data=data
                )

            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Upload successful")
                print(f"  📁 Saved as: {result.get('filename')}")
                print(f"  🔍 VAD Status: {'✅ Passed' if result.get('vad_passed') else '❌ Failed'}")

                if not result.get('vad_passed'):
                    print(f"  ⚠️  VAD Warning: {result.get('vad_message')}")

                return True
            else:
                print(f"  ❌ Upload failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


# ========================================
# BATCH TESTING FUNCTIONS
# ========================================

def run_batch_tests(test_dir, base_url="http://localhost:15018"):
    """Run batch tests on multiple audio files"""
    print(f"\n🧪 BATCH TESTING: {test_dir}")
    print("=" * 50)

    tester = VADSystemTester(base_url)

    # Check system health first
    if not tester.test_health():
        print("❌ System not healthy, aborting batch tests")
        return False

    # Find audio files
    audio_extensions = ['*.wav', '*.mp3', '*.mp4', '*.flac', '*.ogg']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(test_dir, ext)))
        audio_files.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))

    if not audio_files:
        print(f"❌ No audio files found in {test_dir}")
        return False

    print(f"📁 Found {len(audio_files)} audio files")

    # Test results
    results = {
        'total': len(audio_files),
        'detection_success': 0,
        'analysis_success': 0,
        'voice_detected': 0,
        'rejected_by_vad': 0,
        'errors': 0
    }

    for i, audio_file in enumerate(audio_files[:10]):  # Limit to 10 files for demo
        print(f"\n--- Test {i + 1}/{min(10, len(audio_files))} ---")

        # Extract username from file path
        rel_path = os.path.relpath(audio_file, test_dir)
        username = os.path.dirname(rel_path).replace(os.sep, '_') or 'unknown'

        # Test detection
        if tester.test_voice_detection(audio_file, username):
            results['detection_success'] += 1
        else:
            results['errors'] += 1

        # Test analysis
        if tester.test_audio_analysis(audio_file, username):
            results['analysis_success'] += 1

        time.sleep(0.5)  # Small delay between requests

    # Print summary
    print(f"\n📊 BATCH TEST SUMMARY")
    print("=" * 30)
    print(f"Total files tested: {min(10, len(audio_files))}")
    print(f"Detection success: {results['detection_success']}")
    print(f"Analysis success: {results['analysis_success']}")
    print(f"Errors: {results['errors']}")

    return results


# ========================================
# DEPLOYMENT UTILITIES
# ========================================

def check_model_files(model_dir="models"):
    """Check if all required model files exist"""
    print(f"\n📁 CHECKING MODEL FILES: {model_dir}")
    print("-" * 40)

    required_files = [
        "vad_model_enhanced_nn.keras",
        "vad_model_cnn.keras",
        "vad_model_attention.keras",
        "vad_model_resnet.keras",
        "label_encoder_ultimate_fixed.pkl",
        "model_suara_ultimate_fixed.pkl"
        "ensemble_models_fixed.pkl",
        "labels_ultimate_fixed.txt",
        "model_suara_ultimate_fixed.tflite"
    ]

    missing_files = []

    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  ❌ {file} - MISSING")
            missing_files.append(file)

    if missing_files:
        print(f"\n⚠️  WARNING: {len(missing_files)} files missing!")
        print("Please ensure training is completed and all models are saved.")
        return False
    else:
        print(f"\n✅ All model files present!")
        return True


def create_docker_compose():
    """Create docker-compose.yml for deployment"""
    docker_compose_content = """version: '3.8'

services:
  vad-voice-api:
    build: .
    ports:
      - "15018:15018"
    volumes:
      - ./models:/app/models
      - ./temp_uploads:/app/temp_uploads
      - ./train_data:/app/train_data
    environment:
      - FLASK_ENV=production
      - MODEL_DIR=/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:15018/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vad-voice-api
    restart: unless-stopped
"""

    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)

    print("✅ Created docker-compose.yml")


def create_dockerfile():
    """Create Dockerfile for deployment"""
    dockerfile_content = """FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    libsndfile1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p temp_uploads train_data

# Expose port
EXPOSE 15018

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:15018/health || exit 1

# Run application
CMD ["python", "app.py"]
"""

    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    print("✅ Created Dockerfile")


def setup_deployment():
    """Setup deployment files"""
    print("\n🚀 SETTING UP DEPLOYMENT")
    print("=" * 40)

    create_dockerfile()
    create_docker_compose()

    # Create nginx config
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream vad_api {
        server vad-voice-api:15018;
    }

    server {
        listen 80;
        client_max_body_size 100M;

        location / {
            proxy_pass http://vad_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""

    with open("nginx.conf", "w") as f:
        f.write(nginx_config)

    print("✅ Created nginx.conf")
    print("\nDeployment files created:")
    print("  - Dockerfile")
    print("  - docker-compose.yml")
    print("  - nginx.conf")
    print("\nTo deploy:")
    print("  1. docker-compose build")
    print("  2. docker-compose up -d")


# ========================================
# MAIN CLI INTERFACE
# ========================================

def main():
    parser = argparse.ArgumentParser(description="VAD Voice Recognition System - Testing & Deployment")
    parser.add_argument('command', choices=['test', 'batch', 'deploy', 'check'],
                        help='Command to execute')
    parser.add_argument('--url', default='http://localhost:15018',
                        help='API base URL (default: http://localhost:15018)')
    parser.add_argument('--file', help='Audio file to test')
    parser.add_argument('--dir', help='Directory for batch testing')
    parser.add_argument('--models', default='models', help='Models directory')
    parser.add_argument('--username', default='test_user', help='Username for testing')

    args = parser.parse_args()

    if args.command == 'test':
        print("🧪 VAD SYSTEM TESTING")
        print("=" * 50)

        tester = VADSystemTester(args.url)

        # System health and info
        tester.test_health()
        tester.test_system_info()

        # Test with audio file if provided
        if args.file:
            tester.test_voice_detection(args.file, args.username)
            tester.test_audio_analysis(args.file, args.username)
            tester.test_training_upload(args.file, args.username)
        else:
            print("\n💡 Use --file to test with specific audio file")

    elif args.command == 'batch':
        if not args.dir:
            print("❌ Please specify --dir for batch testing")
            return

        run_batch_tests(args.dir, args.url)

    elif args.command == 'check':
        check_model_files(args.models)

    elif args.command == 'deploy':
        if check_model_files(args.models):
            setup_deployment()
        else:
            print("❌ Cannot deploy: missing model files")


if __name__ == "__main__":
    main()