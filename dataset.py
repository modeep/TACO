import numpy as np
import torch 
import torchvision
import torchvision.dataset as dset 

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


# input은 wav format, return 은 spectrogram
def wav2spectrogram(wav_path):
    sample_rate, samples = wavfile.read(wav_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)


# input은 문자열, return은 각 자/모음 나열
# Range=> 초성: 0~20, 중성: 21~41, 종성: 42~69, 띄어쓰기 70, NULL 71
def text2label(string):
    digit_dict = {'0': '영', '1': '일', '2':'이', '3':'삼', '4':'사', 
                  '5':'오', '6':'육', '7':'칠', '8':'팔', '9':'구'}
    label = list()
    for i, c in enumerate(string):
        if c == ' ': 
            label.append(70)
        elif c == '.' or c == '?' or c == '!':
            continue
        else:
            if c.isdigit():
                c = digit_dict[c]

            cho, joong, jong = hangul.separate(c)
            label.append(cho)
            label.append(joong + 21)
            if jong: label.append(jong + 42)

    return label


# load text data
# list(filename, text, duration)
def load_text(txt_path):
    f = lambda x: x[:2] + x[3:]
    with open(txt_path, 'rt', encoding='UTF8') as  txt:
        lines = txt.readlines()
        texts = list(map(lambda x: f(x.split('|')), lines))
    return texts


# zero padding 
def pad_sequence(sequences):
    max_length = max([len(seq) for seq in sequences])
    pass # for seq in sequences: 
    #     while
    return None

class DataSet(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform


if __name__ == '__main__':
    transcript_path = 'd:/Repos/TACO/kss/transcript.txt'
    b = [x[1] for x in load_text(transcript_path)] # get all info from script (file_name, char sequence, duration)
    a = [text2label(x) for x in b] # convert label from char sequence to digit number for training
    c = max([len(x) for x in a]) # get max length for padding 

    pad_sequence(a)