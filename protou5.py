import os
import time
import uuid
import threading
import subprocess
import csv
import concurrent.futures
from collections import Counter
from urllib.parse import unquote
import io  # Added for in-memory QR handling

import qrcode
from flask import Flask, request, render_template_string, send_from_directory, send_file
from googleapiclient.discovery import build

# Audio processing imports
import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment

# Spleeter with explicit backend configuration
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
os.environ['SPLEETER_BACKEND'] = 'pytorch'

# Configuration using environment variables
API_KEY = os.getenv('YT_API_KEY', 'AIzaSyAimZgmc5YesKAYje2fqNtUiwFmo9FHylg')
DOWNLOAD_DIR = os.path.join(os.getcwd(), 'youtube_downloads')
CSV_FILE = os.path.join(os.getcwd(), 'songs.csv')
MIX_OUTPUT = os.path.join(os.getcwd(), 'final_mix.mp3')

# Session configuration
INPUT_WINDOW = 90
CYCLE_DELAY = 10

app = Flask(__name__)
youtube = build("youtube", "v3", developerKey=API_KEY)

# Thread-safe session management
active_token = None
token_expiry = 0
session_inputs = []
session_lock = threading.Lock()
qr_cache = {}

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            pass

def init_download_dir():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Modified QR generation to use in-memory storage
def generate_qr(url):
    qr = qrcode.make(url)
    img_io = io.BytesIO()
    qr.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io

# Updated YouTube download function with improved error handling
def download_youtube_as_mp3(url, output_path=DOWNLOAD_DIR):
    try:
        url = unquote(url.strip())
        output_template = os.path.join(output_path, "%(title)s.%(ext)s")
        
        result = subprocess.run([
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', '0',
            '--no-playlist',
            '--output', output_template,
            url
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "[ExtractAudio] Destination:" in line:
                    return True, line.split("Destination: ")[1].strip()
            return True, "Download completed"
        return False, result.stderr or "Unknown error"

    except Exception as e:
        return False, str(e)

# Modified session controller for cloud compatibility
def session_controller():
    global active_token, token_expiry, session_inputs
    while True:
        active_token = str(uuid.uuid4())
        token_expiry = time.time() + INPUT_WINDOW
        qr_url = f"{os.getenv('BASE_URL', 'http://localhost:5000')}/search/{active_token}"
        
        # Store QR in memory
        qr_cache[active_token] = generate_qr(qr_url)
        
        print(f"Session started: {qr_url}")
        time.sleep(INPUT_WINDOW)
        
        with session_lock:
            top_queries = [q for q, _ in Counter(session_inputs).most_common(3)]
            session_inputs = []
        
        if top_queries:
            process_queries(top_queries)
            files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.mp3')]
            if len(files) >= 3:
                try:
                    tracks = [os.path.join(DOWNLOAD_DIR, f) for f in files[:3]]
                    dj_mix_three_songs(*tracks, MIX_OUTPUT)
                except Exception as e:
                    print(f"Mixing error: {str(e)}")
        
        time.sleep(CYCLE_DELAY)

# Updated Flask routes
@app.route('/qr/<token>')
def serve_qr(token):
    img_io = qr_cache.get(token)
    if img_io:
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    return "QR expired", 404

@app.route('/')
def home():
    return render_template_string('''
        <h1>AI Music Mixer</h1>
        <p>Current session: {% if active_token %}
            <a href="/search/{{ active_token }}">Join Session</a>
        {% else %}Starting soon...{% endif %}</p>
    ''', active_token=active_token)

if __name__ == '__main__':
    init_csv()
    init_download_dir()
    threading.Thread(target=session_controller, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)