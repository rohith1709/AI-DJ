import os
import time
import uuid
import threading
import subprocess
import csv
import concurrent.futures
from collections import Counter
from urllib.parse import unquote

import qrcode
from flask import Flask, request, render_template_string, send_from_directory
from googleapiclient.discovery import build

# Additional imports for audio processing and DJ mixing
import numpy as np
import librosa
import soundfile as sf
import io
import pyrubberband as pyrb
from pydub import AudioSegment

# Spleeter imports (ensure you have a compatible Spleeter installation)
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

#############################################
# CONFIGURATION & INITIAL SETUP
#############################################
# YouTube Data API key – replace with your actual key.
API_KEY = 'AIzaSyAimZgmc5YesKAYje2fqNtUiwFmo9FHylg'
# Directory where downloaded MP3s will be stored
DOWNLOAD_DIR = 'youtube_downloads'
# CSV log for downloaded song URLs
CSV_FILE = 'songs.csv'
# Final mix output file name
MIX_OUTPUT = 'final_mix.mp3'

# Session configuration for QR input (in seconds)
INPUT_WINDOW = 90     # How long a QR session lasts
CYCLE_DELAY = 10      # Delay between sessions

# Initialize Flask (using an older version if needed)
app = Flask(__name__)
youtube = build("youtube", "v3", developerKey=API_KEY)

# Global variables to manage session state
active_token = None
token_expiry = 0
session_inputs = []
session_lock = threading.Lock()

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            pass

def init_download_dir():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

#############################################
# YOUTUBE SEARCH & DOWNLOAD FUNCTIONS
#############################################
def get_youtube_top_result(query):
    """Search YouTube for the top video matching the query."""
    try:
        req = youtube.search().list(
            part='snippet',
            q=query,
            type='video',
            maxResults=1
        )
        resp = req.execute()
        if resp.get('items'):
            video_id = resp['items'][0]['id']['videoId']
            return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print(f"Error searching YouTube: {e}")
    return None

def save_to_csv(video_url):
    """Append the video URL to the CSV log."""
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([video_url])

def download_youtube_as_mp3(url, output_path=DOWNLOAD_DIR, index=None, total=None):
    """Download a YouTube video as MP3 using yt-dlp."""
    try:
        url = unquote(url.strip())
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        status_prefix = f"[{index}/{total}] " if index and total else ""
        print(f"{status_prefix}Downloading: {url}")
        output_template = os.path.join(output_path, "%(title)s.%(ext)s")
        result = subprocess.run([
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', '0',
            '--no-playlist',
            '--no-warnings',
            '--no-progress',
            '--concurrent-fragments', '4',
            '--retries', '3',
            '--output', output_template,
            url
        ], capture_output=True, text=True)
        if result.returncode == 0:
            # Optionally parse the output for the downloaded file path
            for line in result.stdout.split('\n'):
                if "[ExtractAudio] Destination:" in line:
                    file_path = line.split("Destination: ")[1].strip()
                    print(f"{status_prefix}Completed: {os.path.basename(file_path)}")
                    return True, file_path
            print(f"{status_prefix}Download completed for {url}")
            return True, "Download complete"
        else:
            error_msg = result.stderr.strip() or "Unknown error"
            print(f"{status_prefix}Failed: {error_msg}")
            return False, error_msg
    except Exception as e:
        print(f"Error during download: {e}")
        return False, str(e)

def process_queries(queries):
    """Process queries: search YouTube, log URL, and download MP3s concurrently."""
    total = len(queries)
    print(f"Processing {total} queries...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as executor:
        futures = {}
        for i, query in enumerate(queries):
            print(f"Searching for query: '{query}'")
            video_url = get_youtube_top_result(query)
            if video_url:
                save_to_csv(video_url)
                future = executor.submit(download_youtube_as_mp3, video_url, DOWNLOAD_DIR, i+1, total)
                futures[future] = (query, video_url)
            else:
                print(f"No result found for: '{query}'")
        for future in concurrent.futures.as_completed(futures):
            query, video_url = futures[future]
            success, info = future.result()
            if success:
                print(f"Download succeeded for '{query}': {info}")
            else:
                print(f"Download failed for '{query}': {info}")

#############################################
# SPLEETER & MIXING FUNCTIONS
#############################################
def spleeter_separate_2stem(input_path, output_dir):
    """Separate audio into vocals and accompaniment using Spleeter."""
    os.environ['SPLEETER_BACKEND'] = 'pytorch'
    separator = Separator('spleeter:2stems', multiprocess=False)
    audio_loader = AudioAdapter.default()
    waveform, sr = audio_loader.load(input_path, sample_rate=None)
    prediction = separator.separate(waveform)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_subdir = os.path.join(output_dir, base_name)
    os.makedirs(out_subdir, exist_ok=True)
    vocals = prediction['vocals']
    accomp = prediction['accompaniment']
    vocals_path = os.path.join(out_subdir, 'vocals.wav')
    accomp_path = os.path.join(out_subdir, 'accompaniment.wav')
    sf.write(vocals_path, vocals, int(sr))
    sf.write(accomp_path, accomp, int(sr))
    return vocals_path, accomp_path

def load_stem_waveform(path):
    """Load a single stem waveform."""
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

def load_audio_for_analysis(path):
    """Load audio for BPM and beat analysis."""
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

def find_bpm_and_beats(y, sr):
    """Detect BPM and beat times."""
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return bpm, beat_times
    except Exception as e:
        print(f"Beat tracking error: {e}")
        return 120, np.array([])

def detect_vocal_sections(vocals_y, sr, min_energy_threshold=0.01, min_section_duration=4.0, max_section_duration=16.0):
    """
    Detect continuous vocal sections with sustained energy, with improved filtering.
    """
    vocal_energy = librosa.feature.rms(y=vocals_y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(vocal_energy)), sr=sr, hop_length=512)
    
    vocal_sections = []
    current_section_start = None
    current_section_duration = 0

    for i, (energy, timestamp) in enumerate(zip(vocal_energy, times)):
        if energy > min_energy_threshold:
            if current_section_start is None:
                current_section_start = timestamp
            current_section_duration += 1/sr
        else:
            if current_section_start is not None:
                # Only add section if it meets duration criteria
                if min_section_duration <= current_section_duration <= max_section_duration:
                    vocal_sections.append((current_section_start, timestamp))
                current_section_start = None
                current_section_duration = 0

    # Handle case if last section is not closed
    if current_section_start is not None and min_section_duration <= current_section_duration <= max_section_duration:
        vocal_sections.append((current_section_start, times[-1]))

    return vocal_sections

def find_safe_transition_point(vocals_y, sr, beat_times, candidate_sec, search_radius=15.0):
    """
    Find a transition point that carefully avoids vocal sections.
    Prioritizes points between vocal sections or at musical phrases.
    """
    vocal_sections = detect_vocal_sections(vocals_y, sr)
    
    # Find bar times near the candidate
    bar_indices = np.arange(0, len(beat_times), 4)  # Assuming 4/4 time signature
    bar_times = beat_times[bar_indices]
    valid_bars = bar_times[(bar_times >= candidate_sec - search_radius) & 
                            (bar_times <= candidate_sec + search_radius)]
    
    # Sort valid bars by proximity to candidate
    valid_bars = sorted(valid_bars, key=lambda x: abs(x - candidate_sec))
    
    # Prioritize transition points
    for bar_time in valid_bars:
        # Check if the bar time is safely between vocal sections
        is_between_vocals = True
        for start, end in vocal_sections:
            # Ensure transition is not during or too close to a vocal section
            if abs(bar_time - start) < 2.0 or abs(bar_time - end) < 2.0 or (start < bar_time < end):
                is_between_vocals = False
                break
        
        # If safe, return this bar time
        if is_between_vocals:
            return bar_time
    
    # Fallback to original candidate if no perfect point found
    return candidate_sec

def determine_transition_point(y, sr, vocals_y, sr_voc, beat_times, min_t=60.0, max_t=120.0):
    """
    Intelligently determine a good transition point considering song duration.
    """
    dur = len(y)/sr
    
    # Adjust candidate transition point based on song duration
    if dur < min_t:
        candidate = dur * 0.6  # Earlier transition for shorter songs
    elif dur > max_t:
        candidate = min(dur * 0.6, max_t)  # Cap at max_t, but prefer 60% mark
    else:
        candidate = dur * 0.6  # Default to 60% of song duration
    
    # Find a vocal-aware safe transition point
    transition_point = find_safe_transition_point(vocals_y, sr_voc, beat_times, candidate)
    
    return transition_point

def apply_outgoing_tempo(y, sr, transition_time, window_sec, src_bpm, tgt_bpm, max_shift=0.05):
    """Adjust tempo of outgoing track near transition point."""
    start_sec = max(0, transition_time - window_sec)
    end_sec = transition_time
    if src_bpm <= 0:
        src_bpm = 1.0
    if tgt_bpm <= 0:
        tgt_bpm = src_bpm
    ratio = tgt_bpm / src_bpm
    # Clamp ratio to prevent extreme tempo changes
    ratio = max(1.0 - max_shift, min(ratio, 1.0 + max_shift))
    
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    if end_sample <= start_sample:
        return y
    segment = y[start_sample:end_sample]
    if segment.dtype not in [np.float32, np.float64]:
        segment = segment.astype(np.float32)
    stretched_segment = pyrb.time_stretch(segment, sr, ratio)
    return np.concatenate([y[:start_sample], stretched_segment, y[end_sample:]])

def np_audio_to_pydub(y, sr):
    """Convert numpy array to pydub AudioSegment."""
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return AudioSegment.from_file(buffer, format='wav')

def crossfade(audioA, audioB, crossfade_duration=2000):
    """Crossfade two audio segments."""
    return audioA.append(audioB, crossfade=crossfade_duration)

def dj_mix_three_songs(song1_path, song2_path, song3_path, output_path, dj_sfx_path=None):
    """
    Create a DJ mix with improved vocal-aware transition detection.
    """
    sep_dir = "spleeter_temp"
    os.makedirs(sep_dir, exist_ok=True)

    # Separate stems for all songs
    s1_voc, s1_inst = spleeter_separate_2stem(song1_path, sep_dir)
    s2_voc, s2_inst = spleeter_separate_2stem(song2_path, sep_dir)
    s3_voc, s3_inst = spleeter_separate_2stem(song3_path, sep_dir)

    # Load audio for analysis
    y1, sr1 = load_audio_for_analysis(song1_path)
    y2, sr2 = load_audio_for_analysis(song2_path)
    y3, sr3 = load_audio_for_analysis(song3_path)
    
    # Load vocal stems
    y1_voc, sr1v = load_stem_waveform(s1_voc)
    y2_voc, sr2v = load_stem_waveform(s2_voc)
    y3_voc, sr3v = load_stem_waveform(s3_voc)

    # Detect BPM and beats
    bpm1, bt1 = find_bpm_and_beats(y1, sr1)
    bpm2, bt2 = find_bpm_and_beats(y2, sr2)
    bpm3, bt3 = find_bpm_and_beats(y3, sr3)

    # Determine transition points for each track
    t1 = determine_transition_point(y1, sr1, y1_voc, sr1v, bt1)
    t2 = determine_transition_point(y2, sr2, y2_voc, sr2v, bt2)

    # Apply tempo matching 
    y1 = apply_outgoing_tempo(y1, sr1, t1, 8.0, bpm1, bpm2)
    y2 = apply_outgoing_tempo(y2, sr2, t2, 8.0, bpm2, bpm3)

    # Convert to pydub segments and crossfade
    s1_seg = np_audio_to_pydub(y1, sr1)
    s2_seg = np_audio_to_pydub(y2, sr2)
    s3_seg = np_audio_to_pydub(y3, sr3)
    
    # Crossfade configuration
    crossfade_ms = 3000
    t1_ms = int(t1 * 1000)
    t2_ms = int(t2 * 1000)
    
    # Prepare first track segment
    s1_part = s1_seg[:t1_ms]
    
    # Crossfade between first and second track
    mix_12 = crossfade(s1_part, s2_seg, crossfade_duration=crossfade_ms)
    
    # Optional: overlay DJ SFX
    if dj_sfx_path:
        sfx1 = AudioSegment.from_file(dj_sfx_path)
        fx1_pos = max(0, t1_ms - crossfade_ms)
        mix_12 = mix_12.overlay(sfx1, position=fx1_pos)
    
    # Prepare mixed segment duration
    mix_12_duration = (t1_ms - crossfade_ms) + t2_ms
    if mix_12_duration < 0:
        mix_12_duration = 0
    mix_12_part = mix_12[:mix_12_duration]
    
    # Additional optional SFX
    if dj_sfx_path:
        sfx2 = AudioSegment.from_file(dj_sfx_path)
        fx2_pos = max(0, mix_12_duration - crossfade_ms)
        mix_12_part = mix_12_part.overlay(sfx2, position=fx2_pos)
    
    # Final crossfade into third track
    final_mix = crossfade(mix_12_part, s3_seg, crossfade_duration=crossfade_ms)
    
    # Export final mix
    final_mix.export(output_path, format="mp3")
    print(f"Final DJ mix saved at {output_path}")
    return output_path

#############################################
# SESSION CONTROLLER (QR & PROCESSING)
#############################################
def session_controller():
    """
    A background thread that periodically creates a new session.
    It generates a QR code, collects user song queries, downloads MP3s,
    and attempts to create a final mix from at least three songs.
    """
    global active_token, token_expiry, session_inputs
    while True:
        active_token = str(uuid.uuid4())
        token_expiry = time.time() + INPUT_WINDOW
        qr_url = f"http://localhost:5000/search/{active_token}"
        qr = qrcode.make(qr_url)
        qr_path = f"qr_{active_token}.png"
        qr.save(qr_path)
        print(f"[QR ACTIVE] {qr_url} (Valid for {INPUT_WINDOW} seconds)")
        time.sleep(INPUT_WINDOW)
        print("[SESSION ENDED] Processing inputs...")
        with session_lock:
            top_queries = [q for q, _ in Counter(session_inputs).most_common(3)]
            session_inputs = []
        print(f"Top Queries: {top_queries}")
        if top_queries:
            process_queries(top_queries)
            # After downloads, check if at least three MP3 files exist
            files = [os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR)
                     if f.lower().endswith('.mp3')]
            if len(files) >= 3:
                files.sort(key=lambda x: os.path.getmtime(x))
                song1, song2, song3 = files[:3]
                print("Generating final mix from:", song1, song2, song3)
                try:
                    dj_mix_three_songs(song1, song2, song3, MIX_OUTPUT)
                    print(f"Final mix generated: {MIX_OUTPUT}")
                except Exception as e:
                    print("Error generating mix:", e)
            else:
                print("Not enough songs downloaded to generate a mix.")
        else:
            print("No queries received in this session.")
        print("Session cycle complete. Starting new session in a few seconds...\n")
        time.sleep(CYCLE_DELAY)

#############################################
# FLASK ROUTES
#############################################
# Web form template for entering a song query
@app.route('/background.png')
def serve_background():
    return send_from_directory(os.getcwd(), 'background.png')


form_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Music Mixer</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-image: url('/background.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            height: 100vh;
            margin: 0;
            padding-right: 100px;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        }

        .container {
            background: rgba(0, 0, 0, 0.5);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 12px 24px rgba(0,0,0,0.5);
            backdrop-filter: blur(8px);
            max-width: 400px;
            width: 100%;
            text-align: center;
        }

        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-top: 15px;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.3);
        }

        button {
            background-color: #ff4081;
            color: white;
            padding: 10px 20px;
            margin-top: 15px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

        button:hover {
            background-color: #e91e63;
            transform: scale(1.05);
        }

        .session-info {
            margin-top: 20px;
            font-size: 14px;
        }

        img {
            margin-top: 15px;
            width: 150px;
            height: 150px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        }

        h2 {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        h3 {
            color: #ff5252;
            font-size: 18px;
            font-weight: 600;
        }

    </style>
</head>
<body>
    <div class="container">
        {% if valid %}
            <h2>🎧 AI Music Mixing 🎶</h2>
            <form method="POST">
                <input type="text" name="query" placeholder="Enter song name" required />
                <button type="submit">Mix Now</button>
            </form>
            <div class="session-info">
                <p>⏳ Session ends in <strong>{{ remaining }}</strong> seconds.</p>
                <p>📱 Scan QR code:<br><img src="/qr/{{ token }}" alt="QR Code"></p>
            </div>
        {% else %}
            <h3>🚫 Session ended. Wait for the next session.</h3>
        {% endif %}
    </div>
</body>
</html>
"""
@app.route('/search/<token>', methods=['GET', 'POST'])
def search_page(token):
    global active_token, token_expiry
    now = time.time()
    valid = (token == active_token) and (now < token_expiry)
    remaining = int(token_expiry - now) if valid else 0
    if request.method == 'POST':
        if valid:
            query = request.form.get('query', '').strip()
            if query:
                with session_lock:
                    session_inputs.append(query)
                return f"✅ Submitted: {query}"
            else:
                return "❌ Empty query!"
        else:
            return "❌ Session has ended. Please wait for the next session."
    return render_template_string(form_template, token=token, valid=valid, remaining=remaining)

# Serve QR code images
@app.route('/qr/<token>')
def serve_qr(token):
    filename = f"qr_{token}.png"
    return send_from_directory(os.getcwd(), filename)

@app.route('/')
def home():
    return "Welcome to the Integrated Music Mixing Pipeline!"

#############################################
# MAIN EXECUTION
#############################################
if __name__ == '__main__':
    init_csv()
    init_download_dir()
    # Start the session controller in a background thread.
    threading.Thread(target=session_controller, daemon=True).start()
    # Launch the Flask app (use port 5000)
    app.run(debug=True, port=5000)