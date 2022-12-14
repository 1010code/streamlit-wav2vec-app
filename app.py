import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

#############
import torchaudio
import onnx
import onnxruntime
import numpy as np
import scipy.signal as sps
import os
from pythainlp.util import normalize
from pydub import AudioSegment
import librosa
def _normalize(x):  #
    """You must call this before padding.
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py#L101
    Fork TF to numpy
    """
    # -> (1, seqlen)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))


def remove_adjacent(item):  # code from https://stackoverflow.com/a/3460423
    nums = list(item)
    a = nums[:1]
    for item in nums[1:]:
        if item != a[-1]:
            a.append(item)
    return "".join(a)

resamplers = {
     48000: torchaudio.transforms.Resample(48000,16000),
     44100: torchaudio.transforms.Resample(44100,16000),
     32000: torchaudio.transforms.Resample(32000,16000),
     22050: torchaudio.transforms.Resample(22050,16000),
     16000: torchaudio.transforms.Resample(16000,16000),
     }



def asr(path):
    """
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb
    Fork TF to numpy
    """
    audio_format = 'wav'
    if path.name.endswith('mp3'):
        audio_format = 'mp3'
    data, sampling_rate = torchaudio.load(path, format=audio_format)
    data = resamplers[sampling_rate](data)
    #sampleing_rate = 16000
    #samples = round(len(data) * float(new_rate) / sampling_rate)
    #new_data = sps.resample(data, samples)
    speech = np.array(data, dtype=np.float32)
    speech = _normalize(speech)[None]
    padding = np.zeros((speech.shape[0], AUDIO_MAXLEN - speech.shape[1]))
    speech = np.concatenate([speech, padding], axis=-1).astype(np.float32)
    ort_inputs = {"modelInput": speech}
    print(ort_inputs)
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
    prediction = np.argmax(ort_outs, axis=-1)
    print('prediction')
    print(prediction)
    # Text post processing
    _t1 = "".join([res[i] for i in list(prediction[0][0])])
    return normalize("".join([remove_adjacent(j) for j in _t1.split("[PAD]")]))



@st.cache(allow_output_mutation=True)
def load_model():
    # ????????????
    with open("vocab.json", "r", encoding="utf-8-sig") as f:
        d = eval(f.read())
    model = Path("ars_wav2vec2_large-xlsr-52-tw.onnx")
    if not model.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive("1sKUGYv6IDchKDApmtLbWmj3Zbwk_uJ_U", model)
    ort_session = onnxruntime.InferenceSession('ars_wav2vec2_large-xlsr-52-tw.onnx') # load onnx model
    res = dict((v,k) for k,v in d.items())
    res[69]="[PAD]"
    res[68]="[UNK]"
    return d, res, ort_session
d, res, ort_session = load_model()
st.text("wav2vec ?????????????????????")
#############

st.markdown("""??????  ?????????????????????????????????`.mp3`???`.wav`???????????????????????????`16kHz`?????????""")
st.markdown("""???? ??????????????????[??????](https://drive.google.com/drive/folders/1J6x8dqymeYOUt4lm8Irnb0J1CcHrBpHP?usp=share_link)???""")


# ????????????
uploaded_file = st.file_uploader("Choose an audio file")
if uploaded_file is not None:
    
    input_size = 100000
    new_rate = 16000
    AUDIO_MAXLEN = input_size
    st.markdown(f"Result: {asr(uploaded_file)[:-1]}")
