#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read a CSV of word-level segments (utterance id + start/end time),
run wav2vec2 CTC once per utterance, greedy-decode each segment's
frame window, and write the phone sequence back into the same CSV
as one new column.
"""

import os
import csv
import math
import torch
import librosa
from tqdm import tqdm
from collections import defaultdict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# =========================================================
# CONFIG
# =========================================================

CSV_FILE   = "/path/to/input.csv"
WAV_SCP    = "/path/to/wav.scp"
OUTPUT_CSV = "/path/to/output.csv"

MODEL_PATH = "/path/to/model"

SAMPLE_RATE = 16000

# --- column names in the input CSV ---
UTT_COL   = "utterance_id"
START_COL = "ASR_Start_time"
END_COL   = "ASR_end_time"

# --- column names to create in the output CSV ---
OUT_COL         = "wav2vec_phone_seq"
FRAME_START_COL = "frame_start"
FRAME_END_COL   = "frame_end"
N_FRAMES_COL    = "n_frames"

NEW_COLS = [OUT_COL, FRAME_START_COL, FRAME_END_COL, N_FRAMES_COL]


# =========================================================
# LOAD WAV.SCP
# =========================================================

def load_wav_scp():

    wav_map = {}

    with open(WAV_SCP) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                wav_map[parts[0]] = parts[1]

    return wav_map


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading wav2vec model...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(DEVICE)

model.eval()

torch.set_grad_enabled(False)


# =========================================================
# HELPER FUNCTION  (logic unchanged)
# =========================================================

def decode_segment(
    logits,
    start_sec,
    end_sec,
    total_frames,
    utterance_duration_sec
):
    """
    Decode wav2vec phones from a timestamp segment.

    Returns (phones, frame_start, frame_end).
    frame_start / frame_end are "" when no window could be computed.
    """

    if start_sec is None or end_sec is None:
        return "", "", ""

    frame_duration = utterance_duration_sec / total_frames

    frame_start = math.floor(start_sec / frame_duration)
    frame_end   = math.ceil(end_sec / frame_duration)

    frame_start = max(0, frame_start)
    frame_end   = min(total_frames, frame_end)

    if frame_end <= frame_start:
        return "", frame_start, frame_end

    segment_logits = logits[:, frame_start:frame_end, :]

    if segment_logits.shape[1] == 0:
        return "", frame_start, frame_end

    pred_ids = torch.argmax(segment_logits, dim=-1)

    if pred_ids.numel() == 0:
        return "", frame_start, frame_end

    try:
        phones = processor.batch_decode(
            pred_ids.cpu(),
            skip_special_tokens=True
        )[0]

        phones = " ".join(phones.split())

        return phones, frame_start, frame_end

    except Exception:
        return "", frame_start, frame_end


# =========================================================
# MAIN
# =========================================================

def process():

    wav_map = load_wav_scp()

    with open(CSV_FILE, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        rows = list(reader)

        headers = reader.fieldnames

    output_headers = headers + [c for c in NEW_COLS if c not in headers]

    # -----------------------------------------
    # Defaults so every row has every column
    # -----------------------------------------
    for row in rows:
        for c in NEW_COLS:
            row[c] = ""

    # -----------------------------------------
    # Group rows by utterance -> one forward pass each
    # -----------------------------------------
    utt_to_indices = defaultdict(list)

    for idx, row in enumerate(rows):
        utt_to_indices[row[UTT_COL]].append(idx)

    for utt, indices in tqdm(
        utt_to_indices.items(),
        desc="Processing utterances"
    ):

        if utt not in wav_map:

            print(f"[WARNING] Missing wav.scp entry: {utt}")

            for idx in indices:
                rows[idx][OUT_COL] = "__NO_AUDIO__"

            continue

        wav_path = wav_map[utt]

        if not os.path.exists(wav_path):

            print(f"[WARNING] Missing wav file: {wav_path}")

            for idx in indices:
                rows[idx][OUT_COL] = "__NO_AUDIO__"

            continue

        try:
            speech, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

        except Exception as e:

            print(f"[ERROR] {utt}: {e}")

            for idx in indices:
                rows[idx][OUT_COL] = "__LOAD_FAIL__"

            continue

        inputs = processor(
            speech,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(DEVICE)

        with torch.no_grad():
            logits = model(input_values).logits

        total_frames = logits.shape[1]

        utterance_duration_sec = len(speech) / SAMPLE_RATE

        for idx in indices:

            row = rows[idx]

            try:
                start = float(row[START_COL])
                end   = float(row[END_COL])
            except (KeyError, ValueError, TypeError):
                start, end = None, None

            phones, f_start, f_end = decode_segment(
                logits,
                start,
                end,
                total_frames,
                utterance_duration_sec
            )

            row[OUT_COL]         = phones
            row[FRAME_START_COL] = f_start
            row[FRAME_END_COL]   = f_end
            row[N_FRAMES_COL]    = (
                f_end - f_start if f_start != "" else ""
            )

    # =========================================================
    # WRITE OUTPUT CSV
    # =========================================================
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=output_headers)

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\nDone.")
    print(f"Saved: {OUTPUT_CSV}")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    process()
