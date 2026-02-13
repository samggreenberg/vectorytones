import io
import math
import struct
import wave

import random

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
NUM_CLIPS = 20
EMBEDDING_DIM = 128

clips = {}  # id -> {id, duration, file_size, embedding, wav_bytes}
good_votes: set[int] = set()
bad_votes: set[int] = set()


def generate_wav(frequency: float, duration: float) -> bytes:
    """Return raw WAV bytes for a sine-wave tone."""
    num_samples = int(SAMPLE_RATE * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        samples = []
        for i in range(num_samples):
            t = i / SAMPLE_RATE
            value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack("<h", value))
        wf.writeframes(b"".join(samples))
    return buf.getvalue()


def init_clips():
    rng = random.Random(42)
    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)
        embedding = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
        clips[i] = {
            "id": i,
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "embedding": embedding,
            "wav_bytes": wav_bytes,
        }


init_clips()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/clips")
def list_clips():
    result = []
    for c in clips.values():
        result.append(
            {
                "id": c["id"],
                "frequency": c["frequency"],
                "duration": c["duration"],
                "file_size": c["file_size"],
            }
        )
    return jsonify(result)


@app.route("/api/clips/<int:clip_id>/audio")
def clip_audio(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    return send_file(
        io.BytesIO(c["wav_bytes"]),
        mimetype="audio/wav",
        download_name=f"clip_{clip_id}.wav",
    )


@app.route("/api/clips/<int:clip_id>/vote", methods=["POST"])
def vote_clip(clip_id):
    if clip_id not in clips:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(force=True)
    vote = data.get("vote")
    if vote not in ("good", "bad"):
        return jsonify({"error": "vote must be 'good' or 'bad'"}), 400

    if vote == "good":
        if clip_id in good_votes:
            good_votes.discard(clip_id)
        else:
            bad_votes.discard(clip_id)
            good_votes.add(clip_id)
    else:
        if clip_id in bad_votes:
            bad_votes.discard(clip_id)
        else:
            good_votes.discard(clip_id)
            bad_votes.add(clip_id)

    return jsonify({"ok": True})


@app.route("/api/votes")
def get_votes():
    return jsonify(
        {
            "good": sorted(good_votes),
            "bad": sorted(bad_votes),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
