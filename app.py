import os
import re
import csv
import io
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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


def find_torgo_files(root_dir):
    """
    Recursively find all WAV files in a Torgo dataset directory.

    Torgo structure:
      <speaker>/<session>/wav_headMic/<file>.wav
      <speaker>/<session>/wav_arrayMic/<file>.wav
      <speaker>/<session>/prompts/<file>.txt  <- reference text
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith('.wav'):
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
            # Prompts sit in a sibling 'prompts' (or 'Prompts') folder
            session_dir = os.path.dirname(dirpath)
            stem = os.path.splitext(filename)[0]
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
            parts = rel_path.replace('\\', '/').split('/')
            files.append({
                'name': filename,
                'rel_path': rel_path,
                'url': f'/audio_files/{rel_path}',
                'prompt': prompt_text,
                'speaker': parts[0] if len(parts) > 0 else None,
                'session': parts[1] if len(parts) > 1 else None,
                'mic_type': parts[2] if len(parts) > 2 else None,
            })
    return files


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
