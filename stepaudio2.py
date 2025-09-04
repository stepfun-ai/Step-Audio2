import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels
from device_utils import get_device


class StepAudio2Base:

    def __init__(self, model_path: str):
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
        device = get_device()
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        self.eos_token_id = self.llm_tokenizer.eos_token_id

    def __call__(self, messages: list, **kwargs):
        messages, mels = self.apply_chat_template(messages)

        # Tokenize prompts
        prompt_ids = []
        for msg in messages:
            if isinstance(msg, str):
                prompt_ids.append(self.llm_tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"])
            elif isinstance(msg, list):
                prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
            else:
                raise ValueError(f"Unsupported content type: {type(msg)}")
        device = get_device()
        prompt_ids = torch.cat(prompt_ids, dim=-1).to(device)
        attention_mask = torch.ones_like(prompt_ids)

        #mels = None if len(mels) == 0 else torch.stack(mels).cuda()
        #mel_lengths = None if mels is None else torch.tensor([mel.shape[1] - 2 for mel in mels], dtype=torch.int32, device=get_device())
        if len(mels)==0:
            mels = None
            mel_lengths = None
        else:
            mels, mel_lengths = padding_mels(mels)
            device = get_device()
            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)

        generate_inputs = {
            "input_ids": prompt_ids,
            "wavs": mels,
            "wav_lens": mel_lengths,
            "attention_mask":attention_mask
        }

        generation_config = dict(max_new_tokens=2048,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generation_config.update(kwargs)
        generation_config = GenerationConfig(**generation_config)

        outputs = self.llm.generate(**generate_inputs, generation_config=generation_config)
        output_token_ids = outputs[0, prompt_ids.shape[-1] : -1].tolist()
        output_text_tokens = [i for i in output_token_ids if i < 151688]
        output_audio_tokens = [i - 151696 for i in output_token_ids if i > 151695]
        output_text = self.llm_tokenizer.decode(output_text_tokens)
        return output_token_ids, output_text, output_audio_tokens

    def apply_chat_template(self, messages: list):
        results = []
        mels = []
        for msg in messages:
            content = msg
            if isinstance(content, str):
                text_with_audio = content
                results.append(text_with_audio)
            elif isinstance(content, dict):
                if content["type"] == "text":
                    results.append(f"{content['text']}")
                elif content["type"] == "audio":
                    audio = load_audio(content['audio'])
                    for i in range(0, audio.shape[0], 16000 * 25):
                        mel = log_mel_spectrogram(audio[i:i+16000*25], n_mels=128, padding=479)
                        mels.append(mel)
                        audio_tokens = "<audio_patch>" * compute_token_num(mel.shape[1])
                        results.append(f"<audio_start>{audio_tokens}<audio_end>")
                elif content["type"] == "token":
                    results.append(content["token"])
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        # print(results)
        return results, mels


class StepAudio2(StepAudio2Base):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.llm_tokenizer.eos_token = "<|EOT|>"
        self.llm.config.eos_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|EOT|>")
        self.eos_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|EOT|>")

    def apply_chat_template(self, messages: list):
        results = []
        mels = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio = f"<|BOT|>{role}\n{content}"
                text_with_audio += '<|EOT|>' if msg.get('eot', True) else ''
                results.append(text_with_audio)
            elif isinstance(content, list):
                results.append(f"<|BOT|>{role}\n")
                for item in content:
                    if item["type"] == "text":
                        results.append(f"{item['text']}")
                    elif item["type"] == "audio":
                        audio = load_audio(item['audio'])
                        for i in range(0, audio.shape[0], 16000 * 25):
                            mel = log_mel_spectrogram(audio[i:i+16000*25], n_mels=128, padding=479)
                            mels.append(mel)
                            audio_tokens = "<audio_patch>" * compute_token_num(mel.shape[1])
                            results.append(f"<audio_start>{audio_tokens}<audio_end>")
                    elif item["type"] == "token":
                        results.append(item["token"])
                if msg.get('eot', True):
                    results.append('<|EOT|>')
            elif content is None:
                results.append(f"<|BOT|>{role}\n")
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        # print(results)
        return results, mels

if __name__ == '__main__':
    from token2wav import Token2wav

    model = StepAudio2('Step-Audio-2-mini')
    token2wav = Token2wav('Step-Audio-2-mini/token2wav')

    # Text-to-text conversation
    print()
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
            {"role": "assistant", "content": None}
    ]
    tokens, text, _ = model(messages, max_new_tokens=256, temperature=0.7, repetition_penalty=1.05, top_p=0.9, do_sample=True)
    print(text)

    # Text-to-speech conversation
    print()
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
            {"role": "assistant", "content": "<tts_start>", "eot": False}, # Insert <tts_start> for speech response
    ]
    tokens, text, audio = model(messages, max_new_tokens=4096, temperature=0.7, repetition_penalty=1.05, top_p=0.9, do_sample=True)
    print(text)
    print(tokens)
    audio = token2wav(audio, prompt_wav='assets/default_male.wav')
    with open('output-male.wav', 'wb') as f:
        f.write(audio)

    # Speech-to-text conversation
    print()
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
            {"role": "assistant", "content": None}
    ]
    tokens, text, _ = model(messages, max_new_tokens=256, temperature=0.7, repetition_penalty=1.05, top_p=0.9, do_sample=True)
    print(text)

    # Speech-to-speech conversation
    print()
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
            {"role": "assistant", "content": "<tts_start>", "eot": False}, # Insert <tts_start> for speech response
    ]
    tokens, text, audio = model(messages, max_new_tokens=4096, temperature=0.7, repetition_penalty=1.05, top_p=0.9, do_sample=True)
    print(text)
    print(tokens)
    audio = token2wav(audio, prompt_wav='assets/default_female.wav')
    with open('output-female.wav', 'wb') as f:
        f.write(audio)

    # Multi-turn conversation
    print()
    messages.pop(-1)
    messages += [
            {"role": "assistant", "content": [{"type": "text", "text": "<tts_start>"},
                                              {"type": "token", "token": tokens}]},
            {"role": "human", "content": "Now write a 4-line poem about it."},
            {"role": "assistant", "content": None}
    ]
    tokens, text, audio = model(messages, max_new_tokens=256, temperature=0.7, repetition_penalty=1.05, top_p=0.9, do_sample=True)
    print(text)

    # Multi-modal inputs
    print()
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": [{"type": "text", "text": "Translate the speech into Chinese."},
                                          {"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
            {"role": "assistant", "content": None}
    ]
    tokens, text, audio = model(messages, max_new_tokens=256, temperature=0.7, repetition_penalty=1.05, top_p=0.9, do_sample=True)
    print(text)
