import os
import sys

import numpy as np
import pandas as pd
import tqdm

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

import essentia.standard as estd

import soundfile as sf

import subprocess

from pathlib import Path

sr = 44100

def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)


def isolate_vocal(audio_path, sr=sr):
    # Run spleeter on track to remove the background
    separator = Separator('spleeter:2stems')
    audio_loader = AudioAdapter.default()


    waveform, _ = audio_loader.load(audio_path, sample_rate=sr)
    prediction = separator.separate(waveform=waveform)
    clean_vocal = prediction['vocals']

    # Processing
    audio_mono = clean_vocal.sum(axis=1) / 2
    audio_mono_eqloud = estd.EqualLoudness(sampleRate=sr)(audio_mono)

    return audio_mono_eqloud


def get_all_paths(d):
    p = Path(d)
    paths = [str(x).replace(d,'') for x in p.glob('**/*') if '.mp3' in str(x)]
    return paths


def convert_wav_mp3(path):
    # convert
    cmd = f'lame --preset insane "{path}"'
    subprocess.call(cmd, shell=True)

    # delete original
    cmd = f'rm "{path}"'
    subprocess.call(cmd, shell=True)


def main(in_path, out_path):
    try:
        out_path = out_path.replace('mp3', 'wav')
        create_if_not_exists(out_path)

        vocal = isolate_vocal(in_path, sr)
        
        sf.write(out_path, vocal, samplerate=sr)

        convert_wav_mp3(out_path)
        print('...complete!')
    except Exception as e:
        print(f'failed with error {e}')

if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    main(in_path, out_path)