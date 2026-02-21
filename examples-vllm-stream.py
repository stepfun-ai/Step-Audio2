CHUNK_SIZE = 25

def stream_client(model, history, tools, token2wav=None, output_stream=None, prompt_wav='assets/default_female.wav'):
    response = {"tts_content": {"tts_text": '', "tts_audio": ''}, "tool_calls": []}
    buffer = []
    for line, text, audio in model.stream(history, tools=tools, max_tokens=4096, repetition_penalty=1.05, top_p=0.9, temperature=0.7):
        if len(line.get("tool_calls", [])) > 0:
            if len(response["tool_calls"]) == 0:
                response["tool_calls"] += line["tool_calls"]
            else:
                response["tool_calls"][0]['function']['arguments'] = line["tool_calls"][0]['function']['arguments']
        else:
            if text:
                response["tts_content"]["tts_text"] += text
                print(text, end='', flush=True)
            if audio:
                response["tts_content"]["tts_audio"] += line.get("tts_content", {}).get("tts_audio", '')
                print(audio, end='', flush=True)
                if output_stream:
                    buffer += audio
                    if len(buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
                        output = token2wav.stream(buffer[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], prompt_wav=prompt_wav)
                        with open(output_stream, 'ab') as f:
                            f.write(output)
                        buffer = buffer[CHUNK_SIZE:]
    if output_stream and len(buffer) > 0:
        output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True)
        with open(output_stream, 'ab') as f:
            f.write(output)
    return response

if __name__ == "__main__":
    import wave

    from stepaudio2vllm import StepAudio2
    from pathlib import Path
    from token2wav import Token2wav

    api_url = "http://localhost:8000/v1/chat/completions"
    model_name = "step-audio-2-mini"
    prompt_wav = "assets/default_female.wav"

    model = StepAudio2(api_url, model_name)
    token2wav = Token2wav('Step-Audio-2-mini/token2wav')
    tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]
    token2wav.stream(tokens[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], prompt_wav=prompt_wav) # Warm up

    output_stream = Path('output-stream.pcm')
    output_stream.unlink(missing_ok=True)

    history = [
        {"role": "system", "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月28日，星期四\n请用默认女声与用户交流"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/帮我查一下今天上证指数的开盘价是多少.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    tools = [{"type": "function", "function": {"name": "search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": False}}}]

    response = stream_client(model, history, tools, token2wav, output_stream, prompt_wav)

    with open('assets/search_result.txt') as f:
        search_result = f.read().strip()
    history.pop(-1)
    history += [
        {"role": "assistant", "tts_content": response["tts_content"], "tool_calls": response["tool_calls"]},
        {"role": "input", "tool_call_id": response["tool_calls"][0]["id"], "content": [{"type": "text", "text": search_result}, {"type": "text", "text": '\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。'}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    response = stream_client(model, history, tools, token2wav, output_stream, prompt_wav)

    with open(output_stream, 'rb') as f:
        pcm = f.read()
    wav_path = output_stream.with_suffix('.wav')
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)
