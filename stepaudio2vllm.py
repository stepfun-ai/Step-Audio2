import base64
import json
import re
import io
import wave
import aiohttp
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Tuple, Optional

import requests
from utils import load_audio


class StepAudio2:
    audio_token_re = re.compile(r'<audio_(\d+)>')

    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name

    def __call__(self, messages, **kwargs):
        return next(self.stream(messages, **kwargs, stream=False))

    def stream(self, messages, stream=True, **kwargs):
        headers = {"Content-Type": "application/json"}
        payload = kwargs
        payload["messages"] = self.apply_chat_template(messages)
        payload["model"] = self.model_name
        payload["stream"] = stream
        if (payload["messages"][-1].get("role", None) == "assistant") and (
                payload["messages"][-1].get("content", None) is None):
            payload["messages"].pop(-1)
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        elif payload["messages"][-1].get("eot", True):
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        else:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        with requests.post(self.api_url, headers=headers, json=payload, stream=stream) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line == b'':
                    continue
                line = line.decode('utf-8')[6:] if stream else line.decode('utf-8')
                if line == '[DONE]':
                    break
                line = json.loads(line)['choices'][0]['delta' if stream else 'message']
                text = line.get('tts_content', {}).get('tts_text', None)
                text = text if text else line['content']
                audio = line.get('tts_content', {}).get('tts_audio', None)
                audio = [int(i) for i in StepAudio2.audio_token_re.findall(audio)] if audio else None
                yield line, text, audio

    def process_content_item(self, item):
        if item["type"] == "audio":
            audio_tensor = load_audio(item["audio"], target_rate=16000)
            chunks = []
            for i in range(0, audio_tensor.shape[0], 25 * 16000):
                chunk = audio_tensor[i:i + 25 * 16000]
                if len(chunk.numpy()) == 0:
                    continue
                chunk_int16 = (chunk.numpy().clip(-1.0, 1.0) * 32767.0).astype('int16')
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(chunk_int16.tobytes())
                chunks.append({"type": "input_audio",
                               "input_audio": {"data": base64.b64encode(buf.getvalue()).decode('utf-8'),
                                               "format": "wav"}})
            return chunks
        return [item]

    def apply_chat_template(self, messages):
        out = []
        for m in messages:
            if m["role"] == "human" and isinstance(m["content"], list):
                out.append(
                    {"role": m["role"], "content": [j for i in m["content"] for j in self.process_content_item(i)]})
            else:
                out.append(m)
        return out


class AsyncStepAudio2:
    audio_token_re = re.compile(r'<audio_(\d+)>')

    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Tuple[Dict, Optional[str], Optional[List[int]]]:
        """非流式调用，等价于原 StepAudio2(messages, ...)"""
        async for result in self.stream(messages, stream=False, **kwargs):
            return result
        raise RuntimeError("No response received from model.")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[Tuple[Dict, Optional[str], Optional[List[int]]], None]:
        """异步流式生成器，功能与原 .stream() 对齐"""
        headers = {"Content-Type": "application/json"}
        payload = kwargs.copy()
        payload["messages"] = self.apply_chat_template(messages)
        payload["model"] = self.model_name
        payload["stream"] = stream

        # 与原逻辑完全一致的 continue_final_message / add_generation_prompt 判断
        if (payload["messages"][-1].get("role", None) == "assistant") and (
            payload["messages"][-1].get("content", None) is None
        ):
            payload["messages"].pop(-1)
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        elif payload["messages"][-1].get("eot", True):
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        else:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                response.raise_for_status()

                if not stream:
                    # 非流式：读取完整响应
                    raw_text = await response.text()
                    try:
                        data = json.loads(raw_text)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON response: {raw_text[:200]}...") from e

                    msg = data['choices'][0]['message']
                    text = msg.get('tts_content', {}).get('tts_text') or msg.get('content')
                    audio_str = msg.get('tts_content', {}).get('tts_audio')
                    audio = [int(x) for x in self.audio_token_re.findall(audio_str)] if audio_str else None
                    yield msg, text, audio
                else:
                    # 流式：解析 SSE
                    async for line_bytes in response.content:
                        line_str = line_bytes.decode('utf-8').strip()
                        if line_str == '':
                            continue
                        if line_str == 'data: [DONE]':
                            break
                        if line_str.startswith('data: '):
                            json_str = line_str[6:]
                            try:
                                chunk = json.loads(json_str)
                            except json.JSONDecodeError:
                                continue
                            delta = chunk['choices'][0]['delta']
                            text = delta.get('tts_content', {}).get('tts_text') or delta.get('content')
                            audio_str = delta.get('tts_content', {}).get('tts_audio')
                            audio = [int(x) for x in self.audio_token_re.findall(audio_str)] if audio_str else None
                            yield delta, text, audio

    def process_content_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """与原类完全相同的音频分块逻辑"""
        if item["type"] == "audio":
            audio_tensor = load_audio(item["audio"], target_rate=16000)
            chunks = []
            for i in range(0, audio_tensor.shape[0], 25 * 16000):
                chunk = audio_tensor[i:i + 25 * 16000]
                if chunk.numel() == 0:
                    continue
                chunk_np = chunk.numpy().clip(-1.0, 1.0)
                chunk_int16 = (chunk_np * 32767.0).astype('int16')
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(chunk_int16.tobytes())
                wav_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                chunks.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_b64,
                        "format": "wav"
                    }
                })
            return chunks
        return [item]

    def apply_chat_template(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """与原类完全相同的模板应用逻辑"""
        out = []
        for m in messages:
            if m["role"] == "human" and isinstance(m["content"], list):
                expanded = [j for i in m["content"] for j in self.process_content_item(i)]
                out.append({"role": m["role"], "content": expanded})
            else:
                out.append(m)
        return out


if __name__ == "__main__":
    from token2wav import Token2wav

    model = StepAudio2("http://localhost:8000/v1/chat/completions", "step-audio-2-mini")
    token2wav = Token2wav('Step-Audio-2-mini/token2wav')

    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False
    }
    # Text-to-text conversation
    print()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
        {"role": "assistant", "content": None}
    ]
    response, text, _ = model(messages, **sampling_params)
    print(text)

    # Text-to-speech conversation
    print()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
        {"role": "assistant", "content": "<tts_start>", "eot": False},  # Insert <tts_start> for speech response
    ]
    response, text, audio = model(messages, **sampling_params)
    print(text)
    print(audio)
    audio = token2wav(audio, prompt_wav='assets/default_male.wav')
    with open('output-male.wav', 'wb') as f:
        f.write(audio)

    # Speech-to-text conversation
    print()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human",
         "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
        {"role": "assistant", "content": None}
    ]
    response, text, _ = model(messages, **sampling_params)
    print(text)

    # Speech-to-speech conversation
    print()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human",
         "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},  # Insert <tts_start> for speech response
    ]
    response, text, audio = model(messages, **sampling_params)
    print(text)
    print(audio)
    audio = token2wav(audio, prompt_wav='assets/default_female.wav')
    with open('output-female.wav', 'wb') as f:
        f.write(audio)

    # Multi-turn conversation
    print()
    messages.pop(-1)
    messages += [
        {"role": "assistant", "tts_content": response["tts_content"]},
        {"role": "human", "content": "Now write a 4-line poem about it."},
        {"role": "assistant", "content": None}
    ]
    response, text, audio = model(messages, **sampling_params)
    print(text)

    # Multi-modal inputs
    print()
    messages = [
        {"role": "system",
         "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."},
        {"role": "human",
         "content": [{"type": "audio", "audio": "assets/mmau_test.wav"},  # Audio will be always put before text
                     {"type": "text",
                      "text": f"Which of the following best describes the male vocal in the audio? Please choose the answer from the following options: [Soft and melodic, Aggressive and talking, High-pitched and singing, Whispering] Output the final answer in <RESPONSE> </RESPONSE>."}]},
        {"role": "assistant", "content": None}
    ]
    response, text, audio = model(messages, **sampling_params)
    print(text)
