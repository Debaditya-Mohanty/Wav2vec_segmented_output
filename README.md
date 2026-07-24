# Segment-Level wav2vec2 Phone Sequence Extraction

`extract_wav2vec_phone_seq.py`

Takes a CSV of word-level time boundaries, runs a fine-tuned wav2vec2 CTC phone
recognizer over each audio file, and writes back the greedy-decoded phone
sequence falling inside each word's time window — one row of the CSV, one phone
sequence.

---

## 1. Functioning of the scripts


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
   appended.



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
survive.

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
| `frame_start` | int | First logit frame included |
| `frame_end` | int | Last logit frame|
| `n_frames` | int | `frame_end - frame_start` |



### 3.1 Reading empty and sentinel values

| What you see | What it means | Action |
|---|---|---|
| Phones present | Normal result | — |
| Phones empty, `n_frames` ≥ 1 | Window existed, every frame decoded to blank | Genuine no-output. |
| Phones empty, `n_frames` = 0 | `end <= start`, or start beyond audio end | time stamd isseu or alignment issue |
| `__NO_AUDIO__` | Utterance absent from `wav.scp`, or file not on disk | Path  problem |
| `__LOAD_FAIL__` | `librosa.load()` raised | Corrupt or unreadable audio |


---

## 4. How to run

### 4.1 Configure

Edit the config block at the top of the file. Nothing else needs touching.

```python
CSV_FILE   = "/path/to/input.csv"
WAV_SCP    = "/path/to/wav.scp"
OUTPUT_CSV = "/path/to/output.csv"
MODEL_PATH = "/path/to/wav2vec_model"

SAMPLE_RATE = 16000

UTT_COL   = "utterance_id"
START_COL = "ASR_Start_time"
END_COL   = "ASR_end_time"
```


### 4.2 Execute

```bash
python extract_wav2vec_phone_seq.py
```

GPU is used automatically when `torch.cuda.is_available()`. To pin a specific
device:

```bash
CUDA_VISIBLE_DEVICES=0 python extract_wav2vec_phone_seq.py
```

---

## 5. Design notes


### 5.1 CTC blank is taken to sit at `tokenizer.pad_token_id`

Decoding uses `processor.batch_decode(..., skip_special_tokens=True)`. The
HuggingFace `Wav2Vec2CTCTokenizer` performs CTC collapse by treating the
**pad token** as the blank symbol.

If a model is ever swapped in whose blank index is not the pad token index, the
collapse silently misbehaves: true blanks survive into the output as ordinary
symbols, and repeat-merging happens around the wrong index. 

Verification before using a new model:

```python
print(processor.tokenizer.pad_token, processor.tokenizer.pad_token_id)
print(processor.tokenizer.get_vocab())
```

Confirm the blank index matches `pad_token_id`. If it does not, replace
`batch_decode` with an explicit collapse that takes the blank id as a parameter.


### 5.3 Window boundaries are widened, not tightened

```python
frame_start = math.floor(start_sec / frame_duration)
frame_end   = math.ceil(end_sec  / frame_duration)
```

`floor` on the start and `ceil` on the end expand the window by up to one frame
on each side. Combined with each frame's ~25 ms receptive field, a small amount
of neighboring-phone acoustic evidence bleeds into every window.


### 5.4 Very short segments cannot crash the model

Since slicing happens on logits rather than waveform, the usual wav2vec2
minimum-input-length convolution error cannot occur here. Any positive-duration
window yields at least one frame. A word too short to contain a phone therefore
returns an empty string with `n_frames` ≥ 1 .

One genuine edge case: when `start == end`, whether you get zero frames or one
depends on whether `start / frame_duration` lands exactly on an integer. Filter
zero-duration rows upstream rather than relying on float remainders.



