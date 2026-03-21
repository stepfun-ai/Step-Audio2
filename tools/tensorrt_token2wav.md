# Accelerating StepAudio2 Token2wav with NVIDIA TensorRT

This document provides instructions on how to use NVIDIA TensorRT to accelerate the Token2wav module in StepAudio2 for both offline and streaming inference.

## Preparation

### 1. Install Dependencies

Install the necessary packages using pip. For GPU acceleration with TensorRT, use `onnxruntime-gpu`.

```bash
pip install tensorrt onnxruntime-gpu
```

### 2. Export ONNX Models

You need to export the PyTorch models to ONNX format. There are separate scripts for offline (dynamic batch) and streaming (static batch) modes.

**For Offline Inference:**
```bash
python3 tools/export_onnx_offline_token2wav.py
```

**For Streaming Inference:**
```bash
python3 tools/export_onnx_streaming_token2wav.py
```

## Usage

### Offline Inference

Here is an example of how to use the TensorRT-accelerated Token2wav model for offline inference.

```python
from token2wav import Token2wav
import wave

# The tokens to be converted to speech
tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]

# Initialize Token2wav with TensorRT enabled
token2wav = Token2wav('Step-Audio-2-mini/token2wav', enable_trt=True)

# Generate audio
audio_bytes = token2wav(tokens, 'assets/default_male.wav')

# Save the generated audio to a file
with open('output_offline.wav', 'wb') as f:
    f.write(audio_bytes)
```

### Streaming Inference

For streaming inference, you can process tokens in chunks.

```python
from token2wav import Token2wav
from pathlib import Path
import wave

tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]

# Initialize Token2wav for streaming with TensorRT
token2wav = Token2wav('Step-Audio-2-mini/token2wav', enable_trt=True, streaming=True)

# Process the first chunk of tokens
audio_first_chunk = token2wav.stream(tokens[:25 + token2wav.flow.pre_lookahead_len], prompt_wav='assets/default_male.wav')

# Process the remaining tokens as the last chunk
audio_last_chunk = token2wav.stream(tokens[25 + token2wav.flow.pre_lookahead_len:], prompt_wav='assets/default_male.wav', last_chunk=True)

# Save the streaming output to a PCM file
output_stream = Path('output-stream.pcm')
output_stream.unlink(missing_ok=True)
with open(output_stream, 'wb') as f:
    f.write(audio_first_chunk)
    f.write(audio_last_chunk)

# Convert PCM to WAV
with open(output_stream, 'rb') as f:
    pcm = f.read()
wav_path = output_stream.with_suffix('.wav')
with wave.open(str(wav_path), 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm)

```

## Benchmark

The following benchmark was conducted on an NVIDIA L20 GPU, generating 26 audio clips with a total length of 170 seconds. RTF (Real-Time Factor) is calculated as `Cost Time / Total Audio Length`.

| Method    | Note                                | Cost Time      | RTF     |
|-----------|-------------------------------------|----------------|---------|
| Offline   | batch=1, PyTorch                    | 4.32 seconds   | 0.025   |
| Offline   | batch=1, TensorRT enabled           | 2.09 seconds   | 0.012   |
| Offline   | batch=2, PyTorch                    | 3.77 seconds   | 0.022   |
| Offline   | batch=2, TensorRT enabled           | 1.97 seconds   | 0.012   |
| Streaming | batch=1, chunk_size = 1 second, PyTorch | 20.3 seconds   | 0.119   |
| Streaming | batch=1, chunk_size = 1 second, TensorRT | 12.96 seconds  | 0.076   |
