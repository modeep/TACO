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


if __name__ == '__main__':
    print(load_text('d:/Repos/TACO/kss/transcript.txt'))