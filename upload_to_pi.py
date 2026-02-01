import torch
import logging

exp_dir = "/home/users/ntu/angy0091/scratch/SincQDR-VAD-All/pi-test"
os.makedirs(exp_dir, exist_ok=True)

# logging for debug
log_file = os.path.join(exp_dir, 'pi_test1.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)


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
window_duration = 0.63  # seconds
step_size = 0.08        # seconds (for overlap)
wav_path = "/home/users/ntu/angy0091/scratch/SincQDR_Val_Debug/NO_SPEECH/68.wav"

# ----- load waveform -----
waveform, sr = torchaudio.load(wav_path)  # waveform: [C, T]
waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono if stereo
if sr != sample_rate:
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)

def chunk_waveform(waveform, duration=0.63, step_size=0.08, sample_rate=16000):
    waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform # ensures the dimensions of the waveform is 2
    chunk_samples = int(duration * sample_rate)
    step_samples = int(step_size * sample_rate)
    total_samples = waveform.size(1) # total number of samples in mono audio channel

    chunks = []
    start = 0
    while start + chunk_samples <= total_samples:
        end = start + chunk_samples
        chunk = waveform[:, start:end]
        chunks.append(chunk)
        start += step_samples
    return torch.stack(chunks)

chunks = chunk_waveform(waveform, duration=0.63, step_size=0.08, sample_rate=16000) #chunks: torch.Tensor [num_chunks, 1, chunk_samples]
logging.info(chunks.shape) 

logging.info("Original waveform shape:", waveform.shape)
logging.info("Chunks shape:", chunks.shape)  # should be [num_chunks, 1, 10080]

