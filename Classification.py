# ----------------------------
# Prepare training data from Metadata file-----准备训练数据
# ----------------------------
import pandas as pd
from pathlib import Path
# 定义数据集的下载路径
download_path= Path.cwd()/'UrbanSound8K/UrbanSound8K'

# 读取元数据文件
metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
# 将元数据文件加载到DataFrame中
df = pd.read_csv(metadata_file)
# 预览DataFrame的前5行
df.head()
#print(df.head())

# 通过拼接'fold'和'slice_file_name'列来形成文件的相对路径
df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

# 只保留'relative_path'和'classID'这两列
df = df[['relative_path', 'classID']]
# 预览处理后的DataFrame的前5行
df.head()
#print(df.head())


#从文件中读取音频

import math,random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
import librosa

class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------

  # 导入音频处理的库

  def open(audio_file):
      """
      读取音频文件并返回音频信号和采样率。

      该函数使用librosa库来加载音频文件，提取音频信号（也称为音频时间序列）和采样率。
      音频信号代表了音频的数字化表示，采样率指明了音频信号采样的频率。

      参数:
      audio_file: str - 音频文件的路径。

      返回值:
      tuple - 包含两个元素的元组：(1) 音频信号（numpy数组）；(2) 采样率（整数）。
      """
      # 使用librosa读取音频文件，返回音频信号和采样率
      sig, sr = torchaudio.load(audio_file)
      return (sig, sr)


#转换为两个通道--我们将通过将第一个声道复制到第二个声道将单声道文件转换为立体声。
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1,:]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))

#标准化采样率———————我们必须将所有音频标准化并转换为相同的采样率，以便所有数组都具有相同的尺寸。
  import torchaudio
  import torch

  def resample(aud, newsr):
      sig, sr = aud

      if (sr == newsr):
          # Nothing to do
          return aud

      num_channels = sig.shape[0]
      # Resample first channel
      resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
      if (num_channels > 1):
          # Resample the second channel and merge both channels
          retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
          resig = torch.cat([resig, retwo])

      return ((resig, newsr))

  #调整大小到相同的长度

  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:, :max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)
#时移
  def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

#梅尔光谱图
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

#数据增强——时间和频率屏蔽
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec