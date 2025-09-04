import io

import torch
import torchaudio
import s3tokenizer
import onnxruntime

import torchaudio.compliance.kaldi as kaldi
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
from hyperpyyaml import load_hyperpyyaml
from device_utils import get_device


class Token2wav():

    def __init__(self, model_path, float16=False):
        self.float16 = float16

        device = get_device()
        self.audio_tokenizer = s3tokenizer.load_model(f"{model_path}/speech_tokenizer_v2_25hz.onnx").to(device).eval()

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])

        with open(f"{model_path}/flow.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
            self.flow = configs['flow']
        if float16:
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        device = get_device()
        self.flow.to(device).eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{model_path}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        device = get_device()
        self.hift.to(device).eval()

    def __call__(self, generated_speech_tokens, prompt_wav):
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        device = get_device()
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(mels.to(device), mels_lens.to(device))

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(self.spk_model.run(
            None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
        )[0], device=get_device())

        audio, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        device = get_device()
        prompt_mels = prompt_mel.unsqueeze(0).to(device)
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device=device)

        device = get_device()
        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=device)
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device=device)

        device = get_device()
        if device.type == "mps":
            # MPS has limited autocast support, disable for compatibility
            mel = self.flow.inference(generated_speech_tokens, generated_speech_tokens_lens,
                prompt_speech_tokens, prompt_speech_tokens_lens,
                prompt_mels, prompt_mels_lens, spk_emb, 10)
        elif device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
                mel = self.flow.inference(generated_speech_tokens, generated_speech_tokens_lens,
                    prompt_speech_tokens, prompt_speech_tokens_lens,
                    prompt_mels, prompt_mels_lens, spk_emb, 10)
        else:
            # CPU fallback - no autocast
            mel = self.flow.inference(generated_speech_tokens, generated_speech_tokens_lens,
                prompt_speech_tokens, prompt_speech_tokens_lens,
                prompt_mels, prompt_mels_lens, spk_emb, 10)

        wav, _ = self.hift(speech_feat=mel)
        output = io.BytesIO()
        torchaudio.save(output, wav.cpu(), sample_rate=24000, format='wav')

        return output.getvalue()

if __name__ == '__main__':
    token2wav = Token2wav('/mnt/gpfs/lijingbei/Step-Audio-2-mini/token2wav')

    tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]
    audio = token2wav(tokens, 'assets/default_male.wav')
    with open('assets/give_me_a_brief_introduction_to_the_great_wall.wav', 'wb') as f:
        f.write(audio)
