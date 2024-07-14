from modelscope import snapshot_download
import json
import subprocess
import torch
from IPython.display import Audio, display
import torch
import torchaudio
import lzma
import numpy as np
import pybase16384 as b14
import soundfile
import ChatTTS
import time
import pandas as pd
import concurrent.futures

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')




# 将模型文件download到本地
model_dir = snapshot_download('pzc163/chatTTS',cache_dir='./model_cache/chat_tts')
#model_dir="model_cache"

def get_spk_emb(speaker_index):
    # 从 JSON 文件中读取数据
    with open('slct_voice_240605.json', 'r', encoding='utf-8') as json_file:
        slct_idx_loaded = json.load(json_file)

    # 将包含 Tensor 数据的部分转换回 Tensor 对象
    for key in slct_idx_loaded:
        tensor_list = slct_idx_loaded[key]["tensor"]
        slct_idx_loaded[key]["tensor"] = torch.tensor(tensor_list)

    speak_tensor = slct_idx_loaded[speaker_index]["tensor"]
    return encode_spk_emb(speak_tensor)


def encode_spk_emb(spk_emb: torch.Tensor) -> str:
    with torch.no_grad():
        arr: np.ndarray = spk_emb.to(dtype=torch.float16, device="cpu").numpy()
        s = b14.encode_to_string(
            lzma.compress(
                arr.tobytes(),
                format=lzma.FORMAT_RAW,
                filters=[
                    {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}
                ],
            ),
        )
        del arr
    return s


def infer_text(chat,text,output_file):
    start=time.time()
    wavs = chat.infer(text, params_infer_code=params_infer_code)
    soundfile.write(f"{output_file}", wavs[0][0], 24000)

    waveform, sample_rate = torchaudio.load(f'{output_file}')
    num_frames = waveform.shape[1]
    duration_seconds = num_frames / sample_rate

    num_words, num_chars=count_words_and_chars(text)

    return duration_seconds,time.time()-start,num_words,num_chars
def count_words_and_chars(text):
    words = text.split()
    num_words = len(words)
    num_chars = len(text)
    return num_words, num_chars

def infer_executor(chat,texts):
    file_name_info = []
    duration_seconds_info = []
    time_cost_seconds_info = []

    num_words_info=[]
    num_chars_info=[]
    for index, text in enumerate(texts, start=1):
        file_name = f"output_{index}.wav"
        duration_seconds, time_cost_seconds,num_words, num_chars = infer_text(chat, text, file_name)

        file_name_info.append(file_name)
        duration_seconds_info.append(duration_seconds)
        time_cost_seconds_info.append(time_cost_seconds)
        num_words_info.append(num_words)
        num_chars_info.append(num_chars)

    df = pd.DataFrame({
        "file_name": file_name_info,
        "duration_seconds": duration_seconds_info,
        "time_cost_seconds": time_cost_seconds_info,
        "num_words": num_words_info,
        "num_chard": num_chars_info
    })
    return df

def get_device():
    print(torch.__version__)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # a CUDA device object
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('Using CPU')

    print(torch.cuda.get_device_properties(0))
    return device

def send_result_to_hdfs():
    hdfs_path = "hdfs://haruna/home/byte_suite_ai/yuguosen.2020/chattts/result.csv"
    subprocess.call(["hadoop", "fs", "-put", "result.csv", hdfs_path])

def divide_list(lines,chat,thread_num):
    length = len(lines)
    size, remainder = divmod(length, thread_num)
    indices = [0] + [size * (i + 1) + min(i + 1, remainder) for i in range(thread_num)]
    return [(chat,lines[indices[i]:indices[i + 1]]) for i in range(thread_num)]

if __name__ == '__main__':
    start=time.time()
    speaker_index="11"
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=get_spk_emb(speaker_index),
        temperature=0.0001
    )

    chat = ChatTTS.Chat()
    chat.load(source='custom', custom_path=model_dir, compile=True)
    #chat.load(source='local', compile=True)
    print(f"device type:{chat.device.type}")
    print(f"gpu is is_available:{torch.cuda.is_available()}")

    with open('infer_text.txt') as file:
        lines = [line.rstrip() for line in file]

    infer_executor(chat,lines)

    thread_num=1
    with concurrent.futures.ThreadPoolExecutor() as executor:
        args_list = divide_list(lines,chat,thread_num)
        print(args_list)
        results = executor.map(lambda args: infer_executor(*args), args_list)
    results_list=list(results)
    df=pd.concat(results_list)

    #df=infer_executor(chat,lines)
    df['device_type']=chat.device.type

    df.to_csv('result.csv',index=False)
    print(df[['duration_seconds','time_cost_seconds']])
    send_result_to_hdfs()
    print(f"total cost:{time.time()-start}")

