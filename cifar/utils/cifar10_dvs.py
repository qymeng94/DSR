'''
Based on the SpikingJelly repo: https://github.com/fangwei123456/spikingjelly
'''
from torch.utils.data import Dataset
import os
import numpy as np
import threading
import zipfile
from torchvision.datasets import utils
import torch


class FunctionThread(threading.Thread):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs
    def run(self):
        self.f(*self.args, **self.kwargs)

def integrate_events_to_frames(events, height, width, frames_num=10, split_by='time', normalization=None):
    '''
    * :ref:`API in English <integrate_events_to_frames.__init__-en>`

    .. _integrate_events_to_frames.__init__-cn:

    :param events: 键是{'t', 'x', 'y', 'p'}，值是np数组的的字典
    :param height: 脉冲数据的高度，例如对于CIFAR10-DVS是128
    :param width: 脉冲数据的宽度，例如对于CIFAR10-DVS是128
    :param frames_num: 转换后数据的帧数
    :param split_by: 脉冲数据转换成帧数据的累计方式，允许的取值为 ``'number', 'time'``
    :param normalization: 归一化方法，允许的取值为 ``None, 'frequency', 'max', 'norm', 'sum'``
    :return: 转化后的frames数据，是一个 ``shape = [frames_num, 2, height, width]`` 的np数组

    记脉冲数据为 :math:`E_{i} = (t_{i}, x_{i}, y_{i}, p_{i}), i=0,1,...,N-1`，转换为帧数据 :math:`F(j, p, x, y), j=0,1,...,M-1`。

    若划分方式 ``split_by`` 为 ``'time'``，则

    .. math::

        \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
        j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
        j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases} \\\\
        F(j, p, x, y) & = \\sum_{i = j_{l}}^{j_{r} - 1} \\mathcal{I_{p, x, y}(p_{i}, x_{i}, y_{i})}

    若划分方式 ``split_by`` 为 ``'number'``，则

    .. math::

        j_{l} & = [\\frac{N}{M}] \\cdot j \\\\
        j_{r} & = \\begin{cases} [\\frac{N}{M}] \\cdot (j + 1), & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}\\\\
        F(j, p, x, y) & = \\sum_{i = j_{l}}^{j_{r} - 1} \\mathcal{I_{p, x, y}(p_{i}, x_{i}, y_{i})}

    其中 :math:`\\mathcal{I}` 为示性函数，当且仅当 :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})` 时为1，否则为0。

    若 ``normalization`` 为 ``'frequency'``，

        若 ``split_by`` 为 ``time`` 则

            .. math::
                F_{norm}(j, p, x, y) = \\begin{cases} \\frac{F(j, p, x, y)}{\\Delta T}, & j < M - 1
                \\cr \\frac{F(j, p, x, y)}{\\Delta T + (t_{N-1} - t_{0}) \\bmod M}, & j = M - 1 \\end{cases}

        若 ``split_by`` 为 ``number`` 则

            .. math::
                F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{t_{j_{r}} - t_{j_{l}}}


    若 ``normalization`` 为 ``'max'`` 则

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{\\mathrm{max} F(j, p)}

    若 ``normalization`` 为 ``'norm'`` 则

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y) - \\mathrm{E}(F(j, p))}{\\sqrt{\\mathrm{Var}(F(j, p))}}

    若 ``normalization`` 为 ``'sum'`` 则

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{\\sum_{a, b} F(j, p, a, b)}

    * :ref:`中文API <integrate_events_to_frames.__init__-cn>`

    .. _integrate_events_to_frames.__init__-en:

    :param events: a dict with keys are {'t', 'x', 'y', 'p'} and values are numpy arrays
    :param height: the height of events data, e.g., 128 for CIFAR10-DVS
    :param width: the width of events data, e.g., 128 for CIFAR10-DVS
    :param frames_num: frames number
    :param split_by: how to split the events, can be ``'number', 'time'``
    :param normalization: how to normalize frames, can be ``None, 'frequency', 'max', 'norm', 'sum'``
    :return: the frames data with ``shape = [frames_num, 2, height, width]``

    The events data are denoted by :math:`E_{i} = (t_{i}, x_{i}, y_{i}, p_{i}), i=0,1,...,N-1`, and the converted frames
    data are denoted by :math:`F(j, p, x, y), j=0,1,...,M-1`.

    If ``split_by`` is ``'time'``, then

    .. math::

        \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
        j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
        j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases} \\\\
        F(j, p, x, y) & = \\sum_{i = j_{l}}^{j_{r} - 1} \\mathcal{I_{p, x, y}(p_{i}, x_{i}, y_{i})}

    If ``split_by`` is ``'number'``, then

    .. math::

        j_{l} & = [\\frac{N}{M}] \\cdot j \\\\
        j_{r} & = \\begin{cases} [\\frac{N}{M}] \\cdot (j + 1), & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}\\\\
        F(j, p, x, y) & = \\sum_{i = j_{l}}^{j_{r} - 1} \\mathcal{I_{p, x, y}(p_{i}, x_{i}, y_{i})}

    where :math:`\\mathcal{I}` is the characteristic function，if and only if :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})`,
    this function is identically 1 else 0.

    If ``normalization`` is ``'frequency'``,

        if ``split_by`` is ``time``,

            .. math::
                F_{norm}(j, p, x, y) = \\begin{cases} \\frac{F(j, p, x, y)}{\\Delta T}, & j < M - 1
                \\cr \\frac{F(j, p, x, y)}{\\Delta T + (t_{N-1} - t_{0}) \\bmod M}, & j = M - 1 \\end{cases}

        if ``split_by`` is ``number``,

            .. math::
                F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{t_{j_{r}} - t_{j_{l}}}

    If ``normalization`` is ``'max'``, then

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{\\mathrm{max} F(j, p)}

    If ``normalization`` is ``'norm'``, then

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y) - \\mathrm{E}(F(j, p))}{\\sqrt{\\mathrm{Var}(F(j, p))}}

    If ``normalization`` is ``'sum'``, then

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{\\sum_{a, b} F(j, p, a, b)}
    '''
    frames = np.zeros(shape=[frames_num, 2, height * width])

    # 创建j_{l}和j_{r}
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    if split_by == 'time':
        events['t'] -= events['t'][0]  # 时间从0开始
        assert events['t'][-1] > frames_num
        dt = events['t'][-1] // frames_num  # 每一段的持续时间
        idx = np.arange(events['t'].size)
        for i in range(frames_num):
            t_l = dt * i
            t_r = t_l + dt
            mask = np.logical_and(events['t'] >= t_l, events['t'] < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1 if i < frames_num - 1 else events['t'].size

    elif split_by == 'number':
        di = events['t'].size // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di if i < frames_num - 1 else events['t'].size
    else:
        raise NotImplementedError

    # 开始累计脉冲
    # 累计脉冲需要用bitcount而不能直接相加，原因可参考下面的示例代码，以及
    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
    # height = 3
    # width = 3
    # frames = np.zeros(shape=[2, height, width])
    # events = {
    #     'x': np.asarray([1, 2, 1, 1]),
    #     'y': np.asarray([1, 1, 1, 2]),
    #     'p': np.asarray([0, 1, 0, 1])
    # }
    #
    # frames[0, events['y'], events['x']] += (1 - events['p'])
    # frames[1, events['y'], events['x']] += events['p']
    # print('wrong accumulation\n', frames)
    #
    # frames = np.zeros(shape=[2, height, width])
    # for i in range(events['p'].__len__()):
    #     frames[events['p'][i], events['y'][i], events['x'][i]] += 1
    # print('correct accumulation\n', frames)
    #
    # frames = np.zeros(shape=[2, height, width])
    # frames = frames.reshape(2, -1)
    #
    # mask = [events['p'] == 0]
    # mask.append(np.logical_not(mask[0]))
    # for i in range(2):
    #     position = events['y'][mask[i]] * height + events['x'][mask[i]]
    #     events_number_per_pos = np.bincount(position)
    #     idx = np.arange(events_number_per_pos.size)
    #     frames[i][idx] += events_number_per_pos
    # frames = frames.reshape(2, height, width)
    # print('correct accumulation by bincount\n', frames)

    for i in range(frames_num):
        x = events['x'][j_l[i]:j_r[i]]
        y = events['y'][j_l[i]:j_r[i]]
        p = events['p'][j_l[i]:j_r[i]]
        mask = []
        mask.append(p == 0)
        mask.append(np.logical_not(mask[0]))
        for j in range(2):
            position = y[mask[j]] * height + x[mask[j]]
            events_number_per_pos = np.bincount(position)
            frames[i][j][np.arange(events_number_per_pos.size)] += events_number_per_pos

        if normalization == 'frequency':
            if split_by == 'time':
                if i < frames_num - 1:
                    frames[i] /= dt
                else:
                    frames[i] /= (dt + events['t'][-1] % frames_num)
            elif split_by == 'number':
                    frames[i] /= (events['t'][j_r[i]] - events['t'][j_l[i]])  # 表示脉冲发放的频率

            else:
                raise NotImplementedError

        # 其他的normalization方法，在数据集类读取数据的时候进行通过调用normalize_frame(frames: np.ndarray, normalization: str)
        # 函数操作，而不是在转换数据的时候进行
    return frames.reshape((frames_num, 2, height, width))

def normalize_frame(frames: np.ndarray or torch.Tensor, normalization: str):
    eps = 1e-5  # 涉及到除法的地方，被除数加上eps，防止出现除以0
    for i in range(frames.shape[0]):
        if normalization == 'max':
            frames[i][0] /= max(frames[i][0].max(), eps)
            frames[i][1] /= max(frames[i][1].max(), eps)

        elif normalization == 'norm':
            frames[i][0] = (frames[i][0] - frames[i][0].mean()) / np.sqrt(max(frames[i][0].var(), eps))
            frames[i][1] = (frames[i][1] - frames[i][1].mean()) / np.sqrt(max(frames[i][1].var(), eps))

        elif normalization == 'sum':
            frames[i][0] /= max(frames[i][0].sum(), eps)
            frames[i][1] /= max(frames[i][1].sum(), eps)

        else:
            raise NotImplementedError
    return frames

def convert_events_dir_to_frames_dir(events_data_dir, frames_data_dir, suffix, read_function, height, width,
                                              frames_num=10, split_by='time', normalization=None, thread_num=1, compress=False):
    # 遍历events_data_dir目录下的所有脉冲数据文件，在frames_data_dir目录下生成帧数据文件
    def cvt_fun(events_file_list):
        for events_file in events_file_list:
            frames = integrate_events_to_frames(read_function(events_file), height, width, frames_num, split_by,
                                                normalization)
            if compress:
                frames_file = os.path.join(frames_data_dir,
                                           os.path.basename(events_file)[0: -suffix.__len__()] + '.npz')
                np.savez_compressed(frames_file, frames)
            else:
                frames_file = os.path.join(frames_data_dir,
                                           os.path.basename(events_file)[0: -suffix.__len__()] + '.npy')
                np.save(frames_file, frames)
    events_file_list = utils.list_files(events_data_dir, suffix, True)
    if thread_num == 1:
        cvt_fun(events_file_list)
    else:
        # 多线程加速
        thread_list = []
        block = events_file_list.__len__() // thread_num
        for i in range(thread_num - 1):
            thread_list.append(FunctionThread(cvt_fun, events_file_list[i * block: (i + 1) * block]))
            thread_list[-1].start()
            print(f'thread {i} start, processing files index: {i * block} : {(i + 1) * block}.')
        thread_list.append(FunctionThread(cvt_fun, events_file_list[(thread_num - 1) * block:]))
        thread_list[-1].start()
        print(f'thread {thread_num} start, processing files index: {(thread_num - 1) * block} : {events_file_list.__len__()}.')
        for i in range(thread_num):
            thread_list[i].join()
            print(f'thread {i} finished.')

def extract_zip_in_dir(source_dir, target_dir):
    '''
    :param source_dir: 保存有zip文件的文件夹
    :param target_dir: 保存zip解压后数据的文件夹
    :return: None

    将 ``source_dir`` 目录下的所有*.zip文件，解压到 ``target_dir`` 目录下的对应文件夹内
    '''

    for file_name in os.listdir(source_dir):
        if file_name[-3:] == 'zip':
            with zipfile.ZipFile(os.path.join(source_dir, file_name), 'r') as zip_file:
                zip_file.extractall(os.path.join(target_dir, file_name[:-4]))

class EventsFramesDatasetBase(Dataset):
    @staticmethod
    def get_wh():
        '''
        :return: (width, height)
            width: int
                events或frames图像的宽度
            height: int
                events或frames图像的高度
        :rtype: tuple
        '''
        raise NotImplementedError

    @staticmethod
    def read_bin(file_name: str):
        '''
        :param file_name: 脉冲数据的文件名
        :type file_name: str
        :return: events
            键是{'t', 'x', 'y', 'p'}，值是np数组的的字典
        :rtype: dict
        '''
        raise NotImplementedError

    @staticmethod
    def get_events_item(file_name):
        '''
        :param file_name: 脉冲数据的文件名
        :type file_name: str
        :return: (events, label)
            events: dict
                键是{'t', 'x', 'y', 'p'}，值是np数组的的字典
            label: int
                数据的标签
        :rtype: tuple
        '''
        raise NotImplementedError

    @staticmethod
    def get_frames_item(file_name):
        '''
        :param file_name: 帧数据的文件名
        :type file_name: str
        :return: (frames, label)
            frames: np.ndarray
                ``shape = [frames_num, 2, height, width]`` 的np数组
            label: int
                数据的标签
        :rtype: tuple
        '''
        raise NotImplementedError

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        '''
        :param download_root: 保存下载文件的文件夹
        :type download_root: str
        :param extract_root: 保存解压后文件的文件夹
        :type extract_root: str

        下载数据集到 ``download_root``，然后解压到 ``extract_root``。
        '''
        raise NotImplementedError

    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str, normalization: str or None):
        '''
        :param events_data_dir: 保存脉冲数据的文件夹，文件夹的文件全部是脉冲数据
        :type events_data_dir: str
        :param frames_data_dir: 保存帧数据的文件夹
        :type frames_data_dir: str
        :param frames_num: 转换后数据的帧数
        :type frames_num: int
        :param split_by: 脉冲数据转换成帧数据的累计方式
        :type split_by: str
        :param normalization: 归一化方法
        :type normalization: str or None

        将 ``events_data_dir`` 文件夹下的脉冲数据全部转换成帧数据，并保存在 ``frames_data_dir``。
        转换参数的详细含义，参见 ``integrate_events_to_frames`` 函数。
        '''
        raise NotImplementedError

		
		
		
		
		
		
		
		
		
		
		
		

import numpy as np
import os
from torchvision.datasets import utils
import torch
labels_dict = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}
# https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671
resource = {
    'airplane': ('https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'),
    'automobile': ('https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'),
    'bird': ('https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'),
    'cat': ('https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'),
    'deer': ('https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'),
    'dog': ('https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'),
    'frog': ('https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'),
    'horse': ('https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'),
    'ship': ('https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'),
    'truck': ('https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')
}
# https://github.com/jackd/events-tfds/blob/master/events_tfds/data_io/aedat.py


EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event

def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def load_raw_events(fp,
                    bytes_skip=0,
                    bytes_trim=0,
                    filter_dvs=False,
                    times_first=False):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.fromstring(data, dtype='>u4')
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('---')
        print(data[1:21:2])
        raise ValueError('odd number of data elements')
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    return timestamp, raw_addr


def parse_raw_address(addr,
                      x_mask=x_mask,
                      x_shift=x_shift,
                      y_mask=y_mask,
                      y_shift=y_shift,
                      polarity_mask=polarity_mask,
                      polarity_shift=polarity_shift):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def load_events(
        fp,
        filter_dvs=False,
        # bytes_skip=0,
        # bytes_trim=0,
        # times_first=False,
        **kwargs):
    timestamp, addr = load_raw_events(
        fp,
        filter_dvs=filter_dvs,
        #   bytes_skip=bytes_skip,
        #   bytes_trim=bytes_trim,
        #   times_first=times_first
    )
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity



class CIFAR10DVS(EventsFramesDatasetBase):
    @staticmethod
    def get_wh():
        return 128, 128

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        for key in resource.keys():
            file_name = os.path.join(download_root, key + '.zip')
            if os.path.exists(file_name):
                if utils.check_md5(file_name, resource[key][1]):
                    print(f'extract {file_name} to {extract_root}')
                    utils.extract_archive(file_name, extract_root)
                else:
                    print(f'{file_name} corrupted, re-download...')
                    utils.download_and_extract_archive(resource[key][0], download_root, extract_root,
                                                       filename=key + '.zip',
                                                       md5=resource[key][1])
            else:
                utils.download_and_extract_archive(resource[key][0], download_root, extract_root, filename=key + '.zip',
                                                   md5=resource[key][1])


    @staticmethod
    def read_bin(file_name: str):
        with open(file_name, 'rb') as fp:
            t, x, y, p = load_events(fp,
                        x_mask=0xfE,
                        x_shift=1,
                        y_mask=0x7f00,
                        y_shift=8,
                        polarity_mask=1,
                        polarity_shift=None)
            return {'t': t, 'x': 127 - x, 'y': y, 'p': 1 - p.astype(int)}
        # 原作者的代码可能有一点问题，因此不是直接返回 t x y p

    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str,
                              normalization: str or None):
        width, height = CIFAR10DVS.get_wh()
        thread_list = []
        for key in resource.keys():
            source_dir = os.path.join(events_data_dir, key)
            target_dir = os.path.join(frames_data_dir, key)
            os.mkdir(target_dir)
            print(f'mkdir {target_dir}')
            print(f'convert {source_dir} to {target_dir}')
            thread_list.append(FunctionThread(
                convert_events_dir_to_frames_dir,
                source_dir, target_dir, '.aedat',
                CIFAR10DVS.read_bin, height, width, frames_num, split_by, normalization, 1, True))
            thread_list[-1].start()
            print(f'thread {thread_list.__len__() - 1} start')

        for i in range(thread_list.__len__()):
            thread_list[i].join()
            print(f'thread {i} finished')

    @staticmethod
    def get_frames_item(file_name):
        return torch.from_numpy(np.load(file_name)['arr_0']).float(), labels_dict[file_name.split('_')[-2]]

    @staticmethod
    def get_events_item(file_name):
        return CIFAR10DVS.read_bin(file_name), labels_dict[file_name.split('_')[-2]]

    def __init__(self, root: str, train: bool, split_ratio=0.9, use_frame=True, frames_num=10, split_by='number', normalization='max', transform=None):
        '''
        :param root: 保存数据集的根目录
        :type root: str
        :param train: 是否使用训练集
        :type train: bool
        :param split_ratio: 分割比例。每一类中前split_ratio的数据会被用作训练集，剩下的数据为测试集
        :type split_ratio: float
        :param use_frame: 是否将事件数据转换成帧数据
        :type use_frame: bool
        :param frames_num: 转换后数据的帧数
        :type frames_num: int
        :param split_by: 脉冲数据转换成帧数据的累计方式。``'time'`` 或 ``'number'``
        :type split_by: str
        :param normalization: 归一化方法，为 ``None`` 表示不进行归一化；
                        为 ``'frequency'`` 则每一帧的数据除以每一帧的累加的原始数据数量；
                        为 ``'max'`` 则每一帧的数据除以每一帧中数据的最大值；
                        为 ``norm`` 则每一帧的数据减去每一帧中的均值，然后除以标准差
        :type normalization: str or None

        CIFAR10 DVS数据集，出自 `CIFAR10-DVS: An Event-Stream Dataset for Object Classification <https://www.frontiersin.org/articles/10.3389/fnins.2017.00309/full>`_，
        数据来源于DVS相机拍摄的显示器上的CIFAR10图片。原始数据的下载地址为 https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671。

        关于转换成帧数据的细节，参见 :func:`~spikingjelly.datasets.utils.integrate_events_to_frames`。
        '''
        super().__init__()
        self.transform = transform
        self.train = train
        events_root = os.path.join(root, 'events')
        if os.path.exists(events_root):
            print(f'{events_root} already exists')
        else:
            self.download_and_extract(root, events_root)

        self.use_frame = use_frame
        if use_frame:
            self.normalization = normalization
            if normalization == 'frequency':
                dir_suffix = normalization
            else:
                dir_suffix = None
            frames_root = os.path.join(root, f'frames_num_{frames_num}_split_by_{split_by}_normalization_{dir_suffix}')
            if os.path.exists(frames_root):
                print(f'{frames_root} already exists')
            else:
                os.mkdir(frames_root)
                print(f'mkdir {frames_root}')
                self.create_frames_dataset(events_root, frames_root, frames_num, split_by, normalization)
        self.data_dir = frames_root if use_frame else events_root

        self.file_name = []
        if train:
            index = np.arange(0, int(split_ratio * 1000))
        else:
            index = np.arange(int(split_ratio * 1000), 1000)

        for class_name in labels_dict.keys():
            class_dir = os.path.join(self.data_dir, class_name)
            for i in index:
                if self.use_frame:
                    self.file_name.append(os.path.join(class_dir, 'cifar10_' + class_name + '_' + str(i) + '.npz'))
                else:
                    self.file_name.append(os.path.join(class_dir, 'cifar10_' + class_name + '_' + str(i) + '.aedat'))

    def __len__(self):
        return self.file_name.__len__()


    def __getitem__(self, index):
        if self.use_frame:
            frames, labels = self.get_frames_item(self.file_name[index])
            if self.normalization is not None and self.normalization != 'frequency':
                frames = normalize_frame(frames, self.normalization)
            if self.transform is not None:
                frames = self.transform(frames)
            return frames, labels
        else:
            return self.get_events_item(self.file_name[index])
