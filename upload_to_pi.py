import torch
import logging
import os
import torchaudio
import sys
sys.path.append("/home/users/ntu/angy0091/scratch/SincQDR-VAD-All/SincQDR-VAD")
from model.sincqdrvad import SincQDRVAD

device = torch.device("cpu") 

exp_dir = "/home/users/ntu/angy0091/scratch/SincQDR-VAD-All/pi-test"
os.makedirs(exp_dir, exist_ok=True)

# logging for debug
log_file = os.path.join(exp_dir, 'pi_test5.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

PATCH_SIZE = 8
THRESHOLD = 0.5

# inputs shape for the model: torch.Size([8, 1, 10080])
# sample_duration=0.63, sample_rate=16000

# ----- code for chunking the waveform -----
# step_size = 0.08  # seconds
# duration = 0.63
# time_start = count * step_size
# time_end = time_start + duration
# sample_rate = 16000
# start_sample = int(time_start * sample_rate)
# end_sample = int(time_end * sample_rate)
# chunk = waveform[:, start_sample:end_sample]  # shape [1, 10080]

# ----- parameters -----
sample_rate = 16000
window_duration = 0.63  
step_size = 0.08        
wav_path = "/home/users/ntu/angy0091/scratch/SincQDR_Val_Debug/NO_SPEECH/0a62cbfc-0b2c-47be-a2a6-e0e6adda887f.wav"

# ----- load waveform -----
waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True) 
if sr != sample_rate:
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)

# --- chunking into 0.63s for the model to take in ---
def chunk_waveform(waveform, duration=0.63, step_size=0.08, sample_rate=16000):
    waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform 
    chunk_samples = int(duration * sample_rate)
    step_samples = int(step_size * sample_rate)
    total_samples = waveform.size(1)

    chunks = []
    start = 0
    while start + chunk_samples <= total_samples:
        end = start + chunk_samples
        chunk = waveform[:, start:end]
        chunks.append(chunk)
        start += step_samples
    return torch.stack(chunks)

chunks = chunk_waveform(waveform, duration=0.63, step_size=0.08, sample_rate=16000) #chunks: torch.Tensor [num_chunks, 1, chunk_samples]

logging.info(f"Original waveform shape: {waveform.shape}")
logging.info(f"Chunks shape: {chunks.shape}")  # should be [num_chunks, 1, 10080]

# --- input into the pretrained model ---
model = SincQDRVAD(1, 32, 64, PATCH_SIZE, 2, True).to(device)
checkpoint_path = "/home/users/ntu/angy0091/scratch/SincQDR-VAD-All/model_last_epoch_20260206_2.ckpt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.eval()
all_outputs = []

# --- making sure the batch size is 8 cuz i trained on batch size 8 ---
with torch.no_grad():
    for i in range(0, chunks.size(0), 8):
        batch = chunks[i:i+8]         
        batch = batch.to(device)       
        out = model(batch)             
        all_outputs.append(out.cpu())

outputs = torch.cat(all_outputs, dim=0)
logging.info(f"Final outputs shape: {outputs.shape}")

# --- gets the probability of each chunk ---
probs = torch.sigmoid(outputs).squeeze(1)  
logging.info(f"First 10 VAD probabilities: {probs[:10].tolist()}")

# --- get the (start, end, probability) ---
vad_results = []

for i, p in enumerate(probs):
    t_start = i * step_size
    t_end = t_start + window_duration
    vad_results.append((t_start, t_end, float(p)))

logging.info(f"VAD results (start, end, prob): {vad_results}")

# see the threshold of the probability, then combfine each time interval depending on the probability
def merge_vad_results(vad_results, threshold):
    merged_segments = []

    current_start = None
    current_end = None

    for start, end, prob in vad_results:

        if prob >= threshold:
            if current_start is None:
                current_start = start
                current_end = end
            else:
                current_end = end

        else:
            if current_start is not None:
                merged_segments.append((current_start, current_end))
                current_start = None
                current_end = None

    if current_start is not None:
        merged_segments.append((current_start, current_end))

    return merged_segments

speech_segments = merge_vad_results(vad_results, THRESHOLD)

logging.info(f"Merged speech segments: {speech_segments}")
