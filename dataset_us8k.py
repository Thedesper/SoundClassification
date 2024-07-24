from torch.utils.data import DataLoader, Dataset, random_split
import torch
from Classification import df
from Classification import download_path
from Classification import AudioUtil


# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    """
    SoundDS类继承自Dataset，用于处理音频数据集。

    参数:
    - df: 包含音频文件信息的DataFrame，如文件路径和类别ID。
    - data_path: 音频文件的根目录路径。

    属性:
    - df: 存储DataFrame的副本。
    - data_path: 音频文件根目录的字符串表示。
    - duration: 音频样本的目标持续时间（以毫秒为单位）。
    - sr: 音频的采样率。
    - channel: 音频的声道数。
    - shift_pct: 音频时间移位的百分比。
    """
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        """
        返回数据集的大小，即DataFrame的行数。

        返回:
        - int: 数据集的大小。
        """
        return len(self.df)

    # Get i'th item in dataset
    # ----------------------------
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


