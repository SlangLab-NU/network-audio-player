# Torgo Speech Labeler

A web-based tool for playing and labeling audio files from the [Torgo dysarthric speech dataset](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html). Designed for fast, keyboard-driven speech/silence annotation with progress tracking and CSV export.

## Features

- Load a Torgo dataset directory (full dataset, single speaker, or single session)
- Displays the reference prompt text alongside each audio file
- One-click or keyboard-driven **Speech** / **Silence** labeling
- Auto-advances to the next unlabeled file after each label
- Color-coded playlist with filter (All / Unlabeled / Speech / Silence) and search
- Progress bar showing how many files have been labeled
- Export labels to CSV
- Optional WER log viewer for speech recognition comparison

## Torgo Dataset Structure

The tool expects the standard Torgo layout:

```
<root>/
  <speaker>/          e.g. F01, M02
    <session>/        e.g. Session1, Session2
      wav_headMic/
        0001.wav
        0002.wav
        ...
      wav_arrayMic/
        0001.wav
        ...
      prompts/
        0001.txt      contains the spoken phrase for 0001.wav
        ...
```

The root you enter can be the top-level `Torgo/` directory, a single speaker folder like `F01/`, or a single session like `F01/Session1/`.

## Setup

**Requirements:** Python 3.8+

```bash
pip install -r requirements.txt
```

## Running

```bash
python app.py
```

Then open `http://localhost:3000` in your browser.

For production use with gunicorn:

```bash
gunicorn -w 2 -b 0.0.0.0:3000 app:app
```

Or with Docker:

```bash
docker build -t torgo-labeler .
docker run -p 3000:3000 -v /path/to/your/data:/data torgo-labeler
```

Then point the UI at `/data/Torgo` (or whatever subdirectory you mounted).

## Usage

### 1. Load the dataset

Enter the path to your Torgo data in the header and click **Load Dataset**.

```
/path/to/Torgo
```

The playlist populates in the sidebar and playback starts at the first unlabeled file.

### 2. Label files

Listen to each clip and press:

| Key | Action |
|-----|--------|
| `S` | Label as **Speech** and advance |
| `X` | Label as **Silence / Noise** and advance |
| `→` | Skip (no label) |
| `←` | Go to previous file |
| `Space` | Play / Pause |
| `R` | Replay from start |

You can also use the **Speech** / **Silence / Noise** buttons on screen.

After labeling, the tool automatically jumps to the next unlabeled file. Labeled files are shown with colored dots in the sidebar (green = speech, red = silence).

### 3. Filter and search

Use the filter buttons above the playlist to show only **Unlabeled**, **Speech**, or **Silence** files. Type in the search box to filter by filename or prompt text.

### 4. Export labels

Click **Export Labels (CSV)** to download `torgo_labels.csv`:

```csv
file,speaker,session,mic_type,label,prompt
F01/Session1/wav_headMic/0001.wav,F01,Session1,wav_headMic,speech,Junior had been placed in charge
F01/Session1/wav_headMic/0002.wav,F01,Session1,wav_headMic,silence,
...
```

## WER Log (optional)

The tool can also display Word Error Rate (WER) data from a speech recognition evaluation log. Click **WER Log** at the bottom of the main panel to expand this section. You can upload a log file or point it at a path on the server.

Expected log format:

```
File: /path/to/0001.wav
Reference: Junior had been placed in charge
Prediction: junior had been placed in charge
Reference (normalized): junior had been placed in charge
Prediction (normalized): junior had been placed in charge
Individual WER: 0.0000
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the UI |
| `POST` | `/load_torgo` | Load a Torgo directory. Form field: `torgo_path` |
| `POST` | `/save_label` | Save a label. JSON body: `{"rel_path": "...", "label": "speech"\|"silence"}` |
| `GET` | `/export_labels` | Download all labels as `torgo_labels.csv` |
| `GET` | `/status` | Current server state (dataset, labels, progress) |
| `GET` | `/audio_files/<path>` | Serve an audio file (supports subdirectory paths) |
| `POST` | `/upload_log` | Upload a WER log file. Form field: `log_file` |
| `POST` | `/load_log_from_path` | Load a WER log by server path. Form field: `log_path` |

## Notes

- Labels are stored in memory and lost on server restart. Export frequently.
- The tool serves audio files directly from your filesystem — no files are copied or modified.
- Prompt files are read from the `prompts/` (or `Prompts/`) sibling directory of each `wav_*` folder.
