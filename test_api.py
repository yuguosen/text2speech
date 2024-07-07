import datetime
import os
import zipfile
from io import BytesIO

import requests

host = "127.0.0.1"
port = "9880"


def get_speaker_list():
    method = "speakers_list"

    url = f"http://{host}:{port}/{method}"
    print(url)
    response = requests.get(url)
    print(response.content)


def get_tts():
    #method = "generate_voice"
    method = "generate_voice"
    url = f"http://{host}:{port}/{method}/"


    body = {
        "text": [
            "Post like top YouTubers and accelerate your growth with top-quality hooks. Don't let CapCut hold you back, a great hook is 99% of what makes your video go viral",
            ''' 比如'''
        ],
        "stream": False,
        "speaker_index": 1,
    }

    try:
        response = requests.post(url, json=body)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
            # save files for each request in a different folder
            dt = datetime.datetime.now()
            ts = int(dt.timestamp())
            tgt = f"./output/{ts}/"
            os.makedirs(tgt, 0o755)
            zip_ref.extractall(tgt)
            print("Extracted files into", tgt)

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")


get_tts()
