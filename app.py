import os
import re
import csv
import io
import wave
import audioop
import tempfile
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'torgo_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global state
current_directory = None
current_playlist = []
torgo_root = None
torgo_files = []
labels = {}  # {rel_path: "speech" | "silence"}
SUPPORTED_AUDIO_EXTENSIONS = ('.mp3', '.wav', '.ogg')
parsed_transcription_data = {}


def parse_log_content(log_content):
    """Parses a WER log file to extract transcription data."""
    data = {}
    file_block_pattern = re.compile(
        r"File: (.+\.wav)\s*\n"
        r"Reference: .*\n"
        r"Prediction: .*\n"
        r"Reference \(normalized\): (.+)\s*\n"
        r"Prediction \(normalized\): (.+)\s*\n"
        r"Individual WER: ([\d.]+)"
    )
    for match in file_block_pattern.finditer(log_content):
        filename = os.path.basename(match.group(1))
        data[filename] = {
            "reference_normalized": match.group(2).strip(),
            "prediction_normalized": match.group(3).strip(),
            "wer": float(match.group(4))
        }
    return data


SILENCE_PHONEMES = {
    'h#', 'pau', 'epi', 'sil', 'SIL', 'sp', '#h',
    'pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl',
}


def parse_phn(path):
    """Parse a TIMIT-format .phn file into a list of segment dicts."""
    segments = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                segments.append({
                    'start': int(parts[0]),
                    'end': int(parts[1]),
                    'label': parts[2],
                    'is_speech': parts[2] not in SILENCE_PHONEMES,
                })
    return segments


def _find_phn(wav_dir, session_dir, stem):
    """Return the path to a .phn file for the given stem, or None."""
    candidates = [
        os.path.join(wav_dir, stem + '.phn'),
        os.path.join(session_dir, 'phn', stem + '.phn'),
        os.path.join(session_dir, 'PHN', stem + '.phn'),
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def find_torgo_files(root_dir):
    """
    Recursively find all WAV files in a Torgo dataset directory.

    Torgo structure:
      <speaker>/<session>/wav_headMic/<file>.wav
      <speaker>/<session>/wav_arrayMic/<file>.wav
      <speaker>/<session>/prompts/<file>.txt  <- reference text
      <speaker>/<session>/phn/<file>.phn      <- phoneme boundaries (optional)
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith('.wav'):
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
            session_dir = os.path.dirname(dirpath)
            stem = os.path.splitext(filename)[0]

            # Prompt text
            prompt_text = None
            for prompts_name in ('prompts', 'Prompts'):
                candidate = os.path.join(session_dir, prompts_name, stem + '.txt')
                if os.path.exists(candidate):
                    try:
                        with open(candidate, 'r', encoding='utf-8') as f:
                            prompt_text = f.read().strip()
                    except Exception:
                        pass
                    break

            # PHN file
            phn_path = _find_phn(dirpath, session_dir, stem)

            parts = rel_path.replace('\\', '/').split('/')
            files.append({
                'name': filename,
                'rel_path': rel_path,
                'url': f'/audio_files/{rel_path}',
                'prompt': prompt_text,
                'has_phn': phn_path is not None,
                'speaker': parts[0] if len(parts) > 0 else None,
                'session': parts[1] if len(parts) > 1 else None,
                'mic_type': parts[2] if len(parts) > 2 else None,
            })
    return files


def run_vad(wav_path, aggressiveness=2):
    """
    Run WebRTC VAD on a WAV file.
    Returns a list of {start, end, is_speech} dicts with times in seconds.
    Requires: 16-bit PCM WAV at 8/16/32/48 kHz.
    """
    try:
        import webrtcvad
    except ImportError:
        raise RuntimeError('webrtcvad not installed — run: pip install webrtcvad-wheels')

    vad = webrtcvad.Vad(aggressiveness)

    with wave.open(wav_path, 'rb') as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())

    if width != 2:
        raise ValueError(f'WAV must be 16-bit PCM (got {width * 8}-bit)')
    if rate not in (8000, 16000, 32000, 48000):
        raise ValueError(f'Unsupported sample rate {rate} Hz — need 8/16/32/48 kHz')
    if channels == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    elif channels > 2:
        raise ValueError(f'Unsupported channel count {channels}')

    FRAME_MS = 20
    frame_bytes = int(rate * FRAME_MS / 1000) * 2  # 16-bit = 2 bytes/sample

    # Per-frame VAD labels
    labels = []
    for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        labels.append(vad.is_speech(pcm[i:i + frame_bytes], rate))

    if not labels:
        return []

    total_dur = len(pcm) / (rate * 2)

    # Smooth: fill silence gaps shorter than 300 ms between speech regions
    gap_limit = max(1, int(300 / FRAME_MS))
    i = 0
    while i < len(labels):
        if not labels[i]:
            j = i
            while j < len(labels) and not labels[j]:
                j += 1
            # Bridge gap if it's short and bordered by speech on both sides
            if (j - i) <= gap_limit and i > 0 and j < len(labels):
                for k in range(i, j):
                    labels[k] = True
            i = j + 1
        else:
            i += 1

    # Build segments from label runs
    segments = []
    cur_speech = labels[0]
    cur_start = 0.0
    for idx, is_speech in enumerate(labels):
        if is_speech != cur_speech:
            segments.append({
                'start': round(cur_start, 4),
                'end': round(idx * FRAME_MS / 1000.0, 4),
                'is_speech': cur_speech,
            })
            cur_speech = is_speech
            cur_start = idx * FRAME_MS / 1000.0
    segments.append({'start': round(cur_start, 4), 'end': round(total_dur, 4), 'is_speech': cur_speech})

    return segments


def run_silero_vad(wav_path):
    """
    Run Silero VAD on a WAV file.
    Returns a list of {start, end, is_speech} dicts with times in seconds.
    Loads audio via stdlib wave + numpy to avoid torchaudio's broken read_audio (v2.9+).
    """
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        import torch
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f'Missing dependency: {e}')

    with wave.open(wav_path, 'rb') as wf:
        rate     = wf.getframerate()
        channels = wf.getnchannels()
        width    = wf.getsampwidth()
        pcm      = wf.readframes(wf.getnframes())

    if width != 2:
        raise ValueError(f'WAV must be 16-bit PCM (got {width * 8}-bit)')
    if channels == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    elif channels > 2:
        raise ValueError(f'Unsupported channel count {channels}')
    if rate != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    wav_tensor = torch.from_numpy(samples)

    model = load_silero_vad()
    timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=16000, return_seconds=True)
    return [
        {'start': round(float(t['start']), 4), 'end': round(float(t['end']), 4), 'is_speech': True}
        for t in timestamps
    ]


@app.route('/run_silero_vad', methods=['POST'])
def api_run_silero_vad():
    """Run Silero VAD on a loaded dataset file or an uploaded file."""
    data = request.get_json()
    rel_path = data.get('rel_path')
    uploaded = data.get('uploaded')

    if rel_path:
        root = torgo_root or current_directory
        if not root:
            return jsonify({'error': 'No dataset loaded'})
        wav_path = os.path.join(root, rel_path.replace('\\', '/'))
    elif uploaded:
        wav_path = os.path.join(UPLOAD_DIR, os.path.basename(uploaded))
    else:
        return jsonify({'error': 'No file specified'})

    if not os.path.exists(wav_path):
        return jsonify({'error': f'File not found: {wav_path}'})

    try:
        segments = run_silero_vad(wav_path)
        return jsonify({'segments': segments})
    except Exception as e:
        app.logger.error(f'Silero VAD error on {wav_path}: {e}')
        return jsonify({'error': str(e)})


@app.route('/run_vad', methods=['POST'])
def api_run_vad():
    """Run WebRTC VAD on a loaded dataset file or an uploaded file."""
    data = request.get_json()
    rel_path = data.get('rel_path')
    uploaded = data.get('uploaded')

    if rel_path:
        root = torgo_root or current_directory
        if not root:
            return jsonify({'error': 'No dataset loaded'})
        wav_path = os.path.join(root, rel_path.replace('\\', '/'))
    elif uploaded:
        wav_path = os.path.join(UPLOAD_DIR, os.path.basename(uploaded))
    else:
        return jsonify({'error': 'No file specified'})

    if not os.path.exists(wav_path):
        return jsonify({'error': f'File not found: {wav_path}'})

    try:
        segments = run_vad(wav_path)
        return jsonify({'segments': segments})
    except Exception as e:
        app.logger.error(f'VAD error on {wav_path}: {e}')
        return jsonify({'error': str(e)})


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Accept an audio file (and optional .phn) uploaded directly from the browser."""
    if 'audio' not in request.files or request.files['audio'].filename == '':
        return jsonify({'success': False, 'message': 'No audio file provided.'})

    audio_file = request.files['audio']
    filename = os.path.basename(audio_file.filename)
    if not filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
        return jsonify({'success': False, 'message': f'Unsupported file type. Use: {SUPPORTED_AUDIO_EXTENSIONS}'})

    audio_file.save(os.path.join(UPLOAD_DIR, filename))

    phn_data = None
    if 'phn' in request.files and request.files['phn'].filename != '':
        phn_file = request.files['phn']
        phn_save = os.path.join(UPLOAD_DIR, os.path.splitext(filename)[0] + '.phn')
        phn_file.save(phn_save)
        try:
            segments = parse_phn(phn_save)
            phn_data = {
                'segments': segments,
                'total_samples': segments[-1]['end'] if segments else 0
            }
        except Exception as e:
            app.logger.warning(f'Could not parse uploaded PHN: {e}')

    return jsonify({
        'success': True,
        'file': {
            'name': filename,
            'url': f'/uploaded_audio/{filename}',
            'phn': phn_data,
        }
    })


@app.route('/uploaded_audio/<filename>')
def serve_uploaded_audio(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/audio_files/<path:filename>')
def serve_audio_file(filename):
    """Serves audio files; supports subdirectory paths for the Torgo dataset."""
    root = torgo_root or current_directory
    if root:
        full_path = os.path.join(root, filename)
        if os.path.exists(full_path):
            return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))
    return "File not found", 404


@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# Torgo dataset endpoints
# ---------------------------------------------------------------------------

@app.route('/get_phn/<path:filename>')
def get_phn(filename):
    """Return parsed PHN segments for a WAV file."""
    root = torgo_root or current_directory
    if not root:
        return jsonify({'error': 'No dataset loaded'})

    rel_path = filename.replace('\\', '/')
    wav_path = os.path.join(root, rel_path)
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    wav_dir = os.path.dirname(wav_path)
    session_dir = os.path.dirname(wav_dir)

    phn_path = _find_phn(wav_dir, session_dir, stem)
    if not phn_path:
        return jsonify({'error': 'No PHN file found'})

    try:
        segments = parse_phn(phn_path)
        total_samples = segments[-1]['end'] if segments else 0
        return jsonify({'segments': segments, 'total_samples': total_samples})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/load_torgo', methods=['POST'])
def load_torgo():
    """Load a Torgo dataset from a root directory."""
    global torgo_root, torgo_files
    root = request.form.get('torgo_path', '').strip()
    if not root:
        return jsonify({"success": False, "message": "Please enter a path."})
    if not os.path.isdir(root):
        return jsonify({"success": False, "message": f"Directory not found: {root}"})
    torgo_root = root
    torgo_files = find_torgo_files(root)
    labeled_count = sum(1 for f in torgo_files if f['rel_path'] in labels)
    return jsonify({
        "success": True,
        "message": f"Loaded {len(torgo_files)} audio files.",
        "files": [{**f, 'label': labels.get(f['rel_path'])} for f in torgo_files],
        "total": len(torgo_files),
        "labeled": labeled_count
    })


@app.route('/save_label', methods=['POST'])
def save_label():
    """Save a speech/silence label for a file."""
    data = request.get_json()
    rel_path = data.get('rel_path')
    label = data.get('label')
    if not rel_path or label not in ('speech', 'silence'):
        return jsonify({"success": False, "message": "Invalid label data."})
    labels[rel_path] = label
    labeled_count = sum(1 for f in torgo_files if f['rel_path'] in labels)
    return jsonify({"success": True, "labeled": labeled_count, "total": len(torgo_files)})


@app.route('/export_labels', methods=['GET'])
def export_labels():
    """Export all labels as a CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['file', 'speaker', 'session', 'mic_type', 'label', 'prompt'])
    for f in torgo_files:
        writer.writerow([
            f['rel_path'].replace('\\', '/'),
            f.get('speaker', ''),
            f.get('session', ''),
            f.get('mic_type', ''),
            labels.get(f['rel_path'], ''),
            f.get('prompt', '') or ''
        ])
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment; filename=torgo_labels.csv"}
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.route('/status', methods=['GET'])
def get_status():
    labeled_count = sum(1 for f in torgo_files if f['rel_path'] in labels)
    return jsonify({
        "torgoRoot": torgo_root,
        "torgoFiles": [{**f, 'label': labels.get(f['rel_path'])} for f in torgo_files],
        "labeled": labeled_count,
        "total": len(torgo_files),
        # Legacy fields
        "currentDirectory": current_directory,
        "files_with_info": [],
        "logLoaded": bool(parsed_transcription_data)
    })


# ---------------------------------------------------------------------------
# Legacy WER log endpoints (kept for backward compatibility)
# ---------------------------------------------------------------------------

@app.route('/select_directory', methods=['POST'])
def select_directory():
    global current_directory, current_playlist
    directory_path = request.form.get('directory_path')
    if not directory_path:
        return jsonify({"success": False, "message": "Please enter a directory path."})
    if not os.path.isdir(directory_path):
        return jsonify({"success": False, "message": "Invalid directory path."})
    current_directory = directory_path
    current_playlist = sorted([
        os.path.join(current_directory, f)
        for f in os.listdir(current_directory)
        if f.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)
    ])
    files_with_info = []
    for f_path in current_playlist:
        f_name = os.path.basename(f_path)
        files_with_info.append({
            "name": f_name,
            "url": f'/audio_files/{f_name}',
            "transcription": parsed_transcription_data.get(f_name)
        })
    return jsonify({"success": True, "message": f"Directory selected: {directory_path}", "files_with_info": files_with_info})


@app.route('/upload_log', methods=['POST'])
def upload_log():
    global parsed_transcription_data
    if 'log_file' not in request.files or request.files['log_file'].filename == '':
        return jsonify({"success": False, "message": "No log file provided."})
    try:
        log_content = request.files['log_file'].read().decode('utf-8')
        parsed_transcription_data = parse_log_content(log_content)
        return jsonify({"success": True, "message": f"Parsed {len(parsed_transcription_data)} entries."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


@app.route('/load_log_from_path', methods=['POST'])
def load_log_from_path():
    global parsed_transcription_data
    log_file_path = request.form.get('log_path')
    if not log_file_path or not os.path.isfile(log_file_path):
        return jsonify({"success": False, "message": f"File not found: {log_file_path}"})
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            parsed_transcription_data = parse_log_content(f.read())
        return jsonify({"success": True, "message": f"Parsed {len(parsed_transcription_data)} entries."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
