# Segment-Level wav2vec2 Phone Sequence Extraction

`extract_wav2vec_phone_seq.py`

Takes a CSV of word-level time boundaries, runs a fine-tuned wav2vec2 CTC phone
recognizer over each audio file, and writes back the greedy-decoded phone
sequence falling inside each word's time window — one row of the CSV, one phone
sequence.

---

## 1. What the script actually does

The important detail, stated up front because it is easy to misread:

**The model is run once on the entire utterance. The audio is never cut.**
Segmentation happens on the *logit matrix*, after the forward pass, by slicing a
frame range out of it.

Pipeline per utterance:

1. Look up the utterance ID in `wav.scp` to get an audio path.
2. Load the full waveform at 16 kHz (`librosa`).
3. One forward pass through `Wav2Vec2ForCTC`, producing a logit tensor of shape
   `(1, T, V)` — `T` frames, `V` vocabulary size.
4. For every CSV row belonging to that utterance:
   - convert the row's start/end times into frame indices,
   - slice `logits[:, frame_start:frame_end, :]`,
   - take `argmax` over the vocabulary axis,
   - run the tokenizer's CTC collapse (`processor.batch_decode`) to merge
     repeats and strip blanks,
   - store the resulting phone string plus the frame indices used.
5. After all utterances, write the original CSV back out with four new columns
   appended. Row order and every original column are preserved.

Grouping rows by utterance before decoding is the only real optimization: a
30-word utterance costs one forward pass, not thirty.

### Consequence worth understanding

Because the transformer attends over the whole utterance before slicing, the
frames inside a word's window already encode acoustic context from outside that
window. This is **not** the same as feeding the isolated word audio to the model
and decoding it. For GOP-style scoring and miscue detection that is normally the
behavior you want — it matches how the model saw the data at training time — but
do not describe the output as "the word decoded independently," because it isn't.

---

## 2. Inputs

### 2.1 Input CSV

Any CSV, any number of extra columns. Three columns are required, named by the
constants at the top of the script:

| Constant | Default column name | Meaning |
|---|---|---|
| `UTT_COL` | `utterance_id` | Key into `wav.scp` |
| `START_COL` | `ASR_Start_time` | Word start, **seconds**, float |
| `END_COL` | `ASR_end_time` | Word end, **seconds**, float |

Multiple rows per utterance are expected and handled. Rows do not need to be
sorted or contiguous — grouping is done internally by a `defaultdict`.

Times must be in seconds. If your pipeline emits milliseconds or frame indices,
convert before running; the script will happily produce garbage otherwise
because a value of `1200` is a syntactically valid float.

### 2.2 wav.scp

Standard Kaldi format, one entry per line:

```
utt_id_0001 /absolute/path/to/audio_0001.wav
utt_id_0002 /absolute/path/to/audio_0002.wav
```

Split is on first whitespace only (`maxsplit=1`), so paths containing spaces
survive. Extended `wav.scp` entries that use a pipe command
(`utt_id sox ... - |`) are **not** supported — the script calls
`os.path.exists()` on whatever follows the ID and will skip the utterance.

### 2.3 Model

A local directory readable by both `Wav2Vec2Processor.from_pretrained()` and
`Wav2Vec2ForCTC.from_pretrained()`. It must contain the tokenizer/vocab files,
not just model weights, since decoding depends on the tokenizer.

---

## 3. Outputs

The output CSV is the input CSV plus four columns:

| Column | Type | Meaning |
|---|---|---|
| `wav2vec_phone_seq` | string | Greedy CTC phone sequence, space-separated |
| `frame_start` | int | First logit frame included (inclusive) |
| `frame_end` | int | Last logit frame, exclusive |
| `n_frames` | int | `frame_end - frame_start` |

The frame columns exist for diagnosis. They let you separate "the model emitted
nothing for this word" from "the window handed to the model was empty," which
otherwise look identical in the phone column.

### 3.1 Reading empty and sentinel values

| What you see | What it means | Action |
|---|---|---|
| Phones present | Normal result | — |
| Phones empty, `n_frames` ≥ 1 | Window existed, every frame decoded to blank | Genuine no-output. Deletion / very short word candidate |
| Phones empty, `n_frames` = 0 | `end <= start`, or start beyond audio end | Upstream timestamp bug — fix the aligner, not this script |
| All four columns empty | Start/end cell missing or unparseable as float | Check the input CSV |
| `__NO_AUDIO__` | Utterance absent from `wav.scp`, or file not on disk | Path / manifest problem |
| `__LOAD_FAIL__` | `librosa.load()` raised | Corrupt or unreadable audio |

Delete the two sentinel assignments in `process()` if you prefer plain blanks,
but then the four cases above collapse into one and become indistinguishable.

---

## 4. How to run

### 4.1 Configure

Edit the config block at the top of the file. Nothing else needs touching.

```python
CSV_FILE   = "/path/to/input.csv"
WAV_SCP    = "/path/to/wav.scp"
OUTPUT_CSV = "/path/to/output.csv"
MODEL_PATH = "/path/to/model"

SAMPLE_RATE = 16000

UTT_COL   = "utterance_id"
START_COL = "ASR_Start_time"
END_COL   = "ASR_end_time"
```

If your CSV uses different column names, change these constants. Do not edit the
function bodies — every column reference goes through these names.

### 4.2 Execute

```bash
python extract_wav2vec_phone_seq.py
```

GPU is used automatically when `torch.cuda.is_available()`. To pin a specific
device:

```bash
CUDA_VISIBLE_DEVICES=0 python extract_wav2vec_phone_seq.py
```

Progress is shown per utterance via `tqdm`. Warnings for missing audio print to
stdout as they occur; redirect stderr/stdout to a log if the run is long.

### 4.3 Dependencies

```
torch
transformers
librosa
tqdm
```

Standard library: `os`, `csv`, `math`, `collections`.

---

## 5. Design notes and stated assumptions

These are deliberate choices, not oversights. They are recorded here because
they are invisible in the code and will cause silent, plausible-looking wrong
output if the surrounding conditions change.

### 5.1 CTC blank is assumed to sit at `tokenizer.pad_token_id`

Decoding uses `processor.batch_decode(..., skip_special_tokens=True)`. The
HuggingFace `Wav2Vec2CTCTokenizer` performs CTC collapse by treating the
**pad token** as the blank symbol.

If a model is ever swapped in whose blank index is not the pad token index, the
collapse silently misbehaves: true blanks survive into the output as ordinary
symbols, and repeat-merging happens around the wrong index. There is no
exception, no warning — just subtly wrong phone strings.

Verification before using a new model:

```python
print(processor.tokenizer.pad_token, processor.tokenizer.pad_token_id)
print(processor.tokenizer.get_vocab())
```

Confirm the blank index matches `pad_token_id`. If it does not, replace
`batch_decode` with an explicit collapse that takes the blank id as a parameter.

### 5.2 Frame duration is derived from the audio, not from the model stride

```python
frame_duration = utterance_duration_sec / total_frames
```

The true wav2vec2 hop is 320 samples = 20 ms exactly
(`model.config.inputs_to_logits_ratio / SAMPLE_RATE`). The convolutional
frontend consumes a receptive field at the start, so `total_frames` is slightly
fewer than `duration / 0.02`, which makes the derived `frame_duration` slightly
larger than 20 ms.

Net effect is roughly a one-frame (~20 ms) systematic offset, largely
independent of utterance length. This was measured and accepted as tolerable for
the current task. If sub-frame boundary precision ever matters, switch to the
config-derived stride — but re-derive any thresholds tuned against the current
behavior, because they absorbed this offset.

### 5.3 Window boundaries are widened, not tightened

```python
frame_start = math.floor(start_sec / frame_duration)
frame_end   = math.ceil(end_sec  / frame_duration)
```

`floor` on the start and `ceil` on the end expand the window by up to one frame
on each side. Combined with each frame's ~25 ms receptive field, a small amount
of neighboring-phone acoustic evidence bleeds into every window.

This favors recall over precision: short words are less likely to produce an
empty sequence, at the cost of occasional spurious boundary phones. Flip to
`ceil` on start and `floor` on end for the opposite trade-off — but do it
globally, and note which convention produced any given results file, because the
two are not comparable.

### 5.4 Very short segments cannot crash the model

Since slicing happens on logits rather than waveform, the usual wav2vec2
minimum-input-length convolution error cannot occur here. Any positive-duration
window yields at least one frame. A word too short to contain a phone therefore
returns an empty string with `n_frames` ≥ 1 — see §3.1 — rather than raising.

One genuine edge case: when `start == end`, whether you get zero frames or one
depends on whether `start / frame_duration` lands exactly on an integer. Filter
zero-duration rows upstream rather than relying on float remainders.

### 5.5 Exception handling is deliberately narrow in one place

Timestamp parsing catches only `(KeyError, ValueError, TypeError)`. Anything
else — CUDA OOM, tokenizer failure, disk error — is allowed to propagate and
kill the run. A crashed run is recoverable; a run that silently wrote thousands
of empty cells because a bare `except:` swallowed an OOM is not.

The `try` inside `decode_segment` remains broad, and will return an empty string
on any decoding failure. Tighten it if silent decode failures become a concern.

---

## 6. Suggested first check after a run

```python
import pandas as pd
df = pd.read_csv(OUTPUT_CSV)

print(df["n_frames"].describe())
print((df["wav2vec_phone_seq"].fillna("") == "").sum(), "empty rows")

# do empty outputs cluster at short windows, or are they scattered?
print(df.assign(empty=df["wav2vec_phone_seq"].fillna("") == "")
        .groupby(pd.cut(df["n_frames"], [0, 2, 5, 10, 25, 1000]))["empty"]
        .mean())
```

If empty outputs concentrate in the lowest frame buckets, the cause is segment
length and a duration threshold is the right fix. If they are scattered evenly
across bucket sizes, segment length is not the problem — look at alignment
quality or model/domain mismatch instead.
