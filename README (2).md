# SoundClassification

## 通过Urban Sound 8K数据集，介绍音频深度学习中的声音分类问题，包括数据准备、音频预处理、模型构建和训练等步骤

1.准备数据

```python
import pandas as pd
from pathlib import path
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
```

2.读取并处理数据

```python
#从文件中读取音频
import math,random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
import librosa


class AudioUtil():

  def open(audio_file):
      # 使用librosa读取音频文件，返回音频信号和采样率
      #torchaudio.load().
      #该函数接受类似路径的对象或类似文件的对象作为输入。
      #返回值是音频信号-波形 (Tensor) 和采样率 (int) 的元组
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

```

3.定义数据集

```python
from torch.utils.data import DataLoader, Dataset, random_split
import torch
class SoundDS(Dataset):
    
     def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的一个样本。

        参数:
        - idx: 样本的索引。

        返回:
        - tuple: 包含增强后的声谱图和对应的类ID。
        """
        # 根据索引获取音频文件的相对路径和类别ID
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        # 打开音频文件
        aud = AudioUtil.open(audio_file)
        # 将音频重采样到目标采样率
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        # 将音频转换为目标声道数
        rechan = AudioUtil.rechannel(reaud, self.channel)

        # 将音频裁剪或填充到目标持续时间
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        # 对音频进行时间移位
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        # 计算梅尔频谱图
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        # 对频谱图进行数据增强
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id
```

4.数据加载器dataloader

```python
from torch.utils.data import random_split
data_path=download_path
myds = SoundDS(df, data_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders

# 使用PyTorch的数据加载器（DataLoader）来组织训练数据集
# 参数batch_size指定每个训练批次的样本数量，shuffle为True表示在每个epoch开始时打乱数据顺序
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

# 使用PyTorch的数据加载器来组织验证数据集
# 验证数据集的加载不需要打乱数据顺序，因此shuffle设为False
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
```

5.模型

```python
class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        # 调用父类的构造方法，初始化对象
        super().__init__()
        ## 初始化一个空列表，用于存储卷积层
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        
   def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
```

6.定义训练过程

```python
def training(model, train_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  # Repeat for each epoch
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

  print('Finished Training')

num_epochs=100# Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs)
```