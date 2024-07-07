import json
import torch
import numpy as np
import lzma
import pybase16384 as b14

def get_speaker_list():
    with open('slct_voice_240605.json', 'r', encoding='utf-8') as json_file:
        slct_idx_loaded = json.load(json_file)
    result=[]
    for index in slct_idx_loaded:
        item=slct_idx_loaded[index]
        result.append({
            "index":index,
            "gender":item["gender"],
            "describe":item["describe"]
        })
    return result

def get_spk_emb(speaker_index:int):
    # 从 JSON 文件中读取数据
    with open('slct_voice_240605.json', 'r', encoding='utf-8') as json_file:
        slct_idx_loaded = json.load(json_file)

    # 将包含 Tensor 数据的部分转换回 Tensor 对象
    for key in slct_idx_loaded:
        tensor_list = slct_idx_loaded[key]["tensor"]
        slct_idx_loaded[key]["tensor"] = torch.tensor(tensor_list)

    speak_tensor = slct_idx_loaded[str(speaker_index)]["tensor"]
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
