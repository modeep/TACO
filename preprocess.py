import matplotlib.pyplot as plt
from hanja import hangul
from scipy import signal
from scipy.io import wavfile


# input은 wav format, return 은 spectrogram
def wav2spectrogram(wav_path):
    sample_rate, samples = wavfile.read(wav_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)


# input은 문자열, return은 각 자/모음 나열
# Range=> 초성: 0~20, 중성: 21~41, 종성: 42~69, 띄어쓰기 70
def text2label(string):
    label = list()
    for c in string:
        if c == ' ': 
            label.append(70)
        elif c == '.' or c == '?' or c == '!':
            continue
        else:
            cho, joong, jong = hangul.separate(c)
            label.append(cho)
            label.append(joong + 21)
            if jong: label.append(jong + 42)

    return label


# load text data
def load_text(txt_path):
    with open(txt_path, 'rt', encoding='UTF8') as  txt:
        lines = txt.readlines()
        texts = list(map(lambda x: x.split('|')[1], lines))
    return texts


if __name__ == '__main__':
    print(load_text('d:/Repos/TACO/kss/transcript.txt'))  