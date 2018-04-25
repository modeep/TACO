import numpy as np
import torch 
import torchvision
import torch.nn as nn
import torchvision.datasets as dset 

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from hanja import hangul
from scipy import signal
from scipy.io import wavfile


# input은 wav format, return 은 spectrogram
def wav2spectrogram(wav_path):
    sample_rate, samples = wavfile.read(wav_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)


# input은 문자열, return은 각 자/모음 나열
# Range=> 초성: 0~20, 중성: 21~41, 종성: 42~69, 띄어쓰기 70, NULL 71
def text2label(string):
    digit_dict = {'0': '영', '1': '일', '2':'이', '3':'삼', '4':'사', 
                  '5':'오', '6':'육', '7':'칠', '8':'팔', '9':'구'}
    labels = list()
    for seq in string:
        label = list()
        for i, c in enumerate(seq):
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
        labels.append(label)

    return np.array(labels)


# load text data
# list(filename, text, duration)
def load_text(txt_path):
    f = lambda x: x[:2] + x[3:]
    with open(txt_path, 'rt', encoding='UTF8') as  txt:
        lines = txt.readlines()
        texts = list(map(lambda x: f(x.strip().split('|')), lines))
    return texts


# zero padding 
def pad_sequence(sequences):
    max_length = max([len(seq) for seq in sequences])
    return [seq + [71] * (max_length - len(seq)) for seq in sequences]


class TextDataset(Dataset):
    def __init__(self, data_path):
        data = np.array(load_text(data_path))

        self.file = data[:, 0]
        self.text = pad_sequence(text2label(data[:, 1]))
        self.time = data[:, 2]
        
    def __getitem__(self, index):
        return self.text[index]

    def __len__(self):
        return len(self.time)



if __name__ == '__main__':
    transcript_path = 'kss/transcript.txt'
    
    txt_dataset = TextDataset(transcript_path)
    data_loader = DataLoader(dataset=txt_dataset, 
                             batch_size=32, 
                             shuffle=True,
                             num_workers=2)
    
    print('Dataset making and Loading Success')