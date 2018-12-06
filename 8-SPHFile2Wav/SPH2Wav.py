# wjs0和TIMIT数据集的音频的文件头都是NIST，这种文件称为SPH文件，
# 它们都不是正确格式的wav文件，因此借助开源的SPHFile库先转换为wav文件
from sphfile import SPHFile

sph = SPHFile('SA1.WAV')
print(sph.format)
"""
{'sample_rate': 16000, 'channel_count': 1, 'sample_byte_format': '01', 
'sample_n_bytes': 2, 'sample_sig_bits': 16, 'sample_coding': 'pcm'}
"""
sph.write_wav('converted.wav')