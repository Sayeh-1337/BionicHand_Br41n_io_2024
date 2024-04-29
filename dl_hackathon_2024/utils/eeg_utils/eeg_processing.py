# Copyright (C) 2023 Langlois Quentin, ICTEAM, UCLouvain. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

#from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes, WaveletTypes, DetrendOperations, WindowOperations

from utils.embeddings import embeddings_to_np
from utils.tensorflow_utils import execute_eagerly

eeg_signature      = tf.TensorSpec(shape = (None, None), dtype = tf.float32)
rate_signature     = tf.TensorSpec(shape = (), dtype = tf.int32)
channels_signature = tf.TensorSpec(shape = (None, ), dtype = tf.string)

class EEGNormalization(enum.IntEnum):
    NORMAL  = 1
    MAX     = 2
    MIN_MAX = 3
    TANH    = 4
    GLOBAL_NORMAL  = 5
    GlobalNormal   = 5
    GLOBAL_MIN_MAX = 6
    GlobalMinMax   = 6

_global_normalization_scheme = (EEGNormalization.GLOBAL_NORMAL, EEGNormalization.GLOBAL_MIN_MAX)

def select_channel(data, channel):
    if isinstance(channel, (int, str)): channel = [channel]
    
    is_df = isinstance(data, pd.DataFrame)
    if is_df: data = data.to_dict('records')

    for row in data:
        indexes = [
            ch if isinstance(ch, int) else row['channels'].index(ch) for ch in channel
        ]
        row.update({
            'channels'   : [row['channels'][idx] for idx in indexes],
            'n_channels' : len(indexes)
        })
        if 'eeg' in row: row['eeg'] = row['eeg'][indexes]

    return pd.DataFrame(data) if is_df else data

def get_eeg_samples(eeg, rate, channels, start = -1, end = -1, time = -1, ** _):
    """
        Crop `eeg` in the range [start | end - time : end | start + time[
        
        Arguments :
            - eeg      : `np.ndarray`, `tf.Tensor` or `mne.io.Raw`, the eeg data to crop
            - rate     : the eeg sampling rate (used to normalize start / end / time from time (float) to samples (int))
            - channels : list of channel names (not used)
            - start    : start position in samples (int) or in seconds (float)
            - end      : end position in samples (int) or in seconds (float)
            - time     : number of samples to keep in samples (int) or in seconds (float)
        Return :
            - cropped  : the cropped eeg with same type as `eeg`
    """
    if start < 0 and end < 0 and time < 0: return eeg
    elif start > eeg.shape[1] or end > eeg.shape[1]: return eeg
    
    start, end, time = _to_pytype(start), _to_pytype(end), _to_pytype(time)

    if isinstance(start, float):    start = int(start * rate)
    if isinstance(end, float):      end   = int(end * rate)
    if isinstance(time, float):     time  = int(time * rate)

    if start < 0: start = 0
    if end < 0:   end = 0
    if time > 0:
        if start:   end = start + time
        elif end:   start = max(0, end - time)
        else:       end = time

    if start == end: return np.zeros((len(channels), 0), dtype = np.float32)

    if isinstance(eeg, (np.ndarray, tf.Tensor)):
        if end > 0: eeg = eeg[:,  : end]
        if start:   eeg = eeg[:, start : ]
    else:
        eeg = eeg.crop(tmin = start / rate, tmax = (end / rate) if end else None, include_tmax = False, verbose = False)
    
    return eeg

def get_event_samples(events, offset = 0, time_window = 0, ** _):
    """
        Crops each event in `events` by applying the offset / taking a fixed number of samples
        /!\ This operation modifies `events` inplace ! use `copy.deepcopy(events)` if you need the original version
        
        Arguments :
            - events      : `list` of `dict`, the events as returned by `read_eeg_events(...)`
            - offset      : start position in samples (int) or in seconds (float)
            - time_window : number of samples to keep in samples (int) or in seconds (float)
        Return :
            - events      : simply return the `events` argument modified inplace
    """
    _is_df = False
    if time_window > 0 or offset > 0:
        if isinstance(events, pd.DataFrame): _is_df, events = True, events.to_dict('records')

        time_window, offset = _to_pytype(time_window), _to_pytype(offset)
        for i, event in enumerate(events):
            off_samples = offset if isinstance(offset, int) else int(offset * event['rate'])
            win_samples = time_window if isinstance(time_window, int) else int(time_window * event['rate'])

            start, end = _to_pytype(event['start']), _to_pytype(event['end'])
            
            start = start if isinstance(start, int) else int(event['start'] * event['rate'])
            start = start + off_samples

            end   = end if isinstance(end, int) else int(event['end'] * event['rate'])
            end   = min(end, start + win_samples)
            event.update({
                'start' : start, 'end' : end, 'time' : end - start
            })
            if 'eeg' in event:
                event['eeg'] = event['eeg'][:, off_samples : off_samples + event['time']]
                if event['eeg'].shape[1] == 0:
                    raise RuntimeError('Signal of event #{}/{} becomes empty !\n{}'.format(i + 1, len(events), event))
    
    return events if not _is_df else pd.DataFrame(events)


def convert_eeg_dtype(eeg, dtype, ** _):
    """
        Converts `eeg` dtype by applying the following transformation :
            - floating -> integer : multiply by 1e6
            - integer -> floating : divide by 1e6
    """
    if isinstance(eeg, tf.Tensor):
        if dtype == np.float32: dtype = tf.float32
        elif dtype == np.int32: dtype = tf.int32
        
        if eeg.dtype == tf.int32 and dtype == tf.float32: return tf.cast(eeg / 1e6, tf.float32)
        elif eeg.dtype == tf.float32 and dtype == tf.int32: return tf.cast(eeg * 1e6, tf.int32)
        else: return eeg
    else:
        if np.issubdtype(dtype, np.floating):
            if np.issubdtype(eeg.dtype, np.floating): return eeg.astype(dtype)
            return (eeg / 1e6).astype(dtype)
        if np.issubdtype(eeg.dtype, np.integer): return eeg.astype(dtype)
        return (eeg * 1e6).astype(dtype)
    

def rearrange_channels(eeg, channels, target_channels, ** _):
    if isinstance(target_channels, (int, np.int32)): return eeg[: target_channels], channels[: target_channels]
    if isinstance(eeg, tf.Tensor): eeg = eeg.numpy()
    ch_to_idx = {ch : i for i, ch in enumerate(channels)}
    
    indices   = [ch_to_idx[ch] for ch in target_channels]
    return eeg[indices], target_channels

@execute_eagerly(signature = eeg_signature, numpy = True)
def resample_eeg(eeg, rate, target_rate, ** kwargs):
    """ Calls the `resample` method of `mne.io.Raw` object """
    if isinstance(eeg, tf.Tensor): eeg = eeg.numpy()
    return librosa.resample(eeg, orig_sr = rate, target_sr = target_rate, axis = -1, fix = False)

@execute_eagerly(signature = eeg_signature, numpy = True)
def detrend_eeg(eeg, * args, ** _):
    """ Calls the `brainflow.detrend` method on each channel of `eeg` """
    if isinstance(eeg, tf.Tensor): eeg = eeg.numpy()
    eeg = eeg.astype(np.float64)
    for i in range(len(eeg)):
        DataFilter.detrend(eeg[i], DetrendOperations.LINEAR.value)
    return eeg.astype(np.float32)

@execute_eagerly(signature = eeg_signature, numpy = True)
def filter_eeg(eeg, rate, lowfreq = None, highfreq = None, freqs = None, order = 4, ** _):
    """ Calls the `DataFilter` (brainflow) filtering methods to remove frequencies lower than `lowfreq` and higher than `highfreq` """
    if isinstance(eeg, tf.Tensor): eeg = eeg.numpy()
    eeg, rate = eeg.astype(np.float64), int(rate)
    
    if freqs is not None: lowfreq, highfreq = [int(f) for f in freqs]
    for i, channel in enumerate(eeg):
        if lowfreq: DataFilter.perform_highpass(channel, rate, lowfreq, order, FilterTypes.BUTTERWORTH, 0)
        if highfreq:  DataFilter.perform_lowpass(channel, rate, highfreq, order, FilterTypes.BUTTERWORTH, 1)
        eeg[i] = channel
    return eeg.astype(np.float32)

@execute_eagerly(signature = eeg_signature, numpy = True)
def remove_environmental_noise(eeg, rate, noise_type = None, ** _):
    """ Calls the `brainflow remove_environmental_noise` method on each channel """
    if isinstance(eeg, tf.Tensor): eeg = eeg.numpy()
    eeg = eeg.astype(np.float64)
    for i in range(len(eeg)):
        DataFilter.remove_environmental_noise(eeg[i], int(rate), noise_type)
    return eeg.astype(np.float32)

@execute_eagerly(signature = eeg_signature, numpy = True)
def reduce_eeg_noise(eeg, rate, channels, channels_to_denoise = ['Fp1'], wave_type = None, decomposition_level = 1, ** _):
    if isinstance(eeg, tf.Tensor): eeg = eeg.numpy()
    eeg = eeg.astype(np.float64)
    for i, name in enumerate(channels):
        if name in channels_to_denoise:
            DataFilter.perform_wavelet_denoising(eeg[i], denoise_wave_type, decomposition_level)
    return eeg.astype(np.float32)


def normalize_eeg(eeg, normalize = EEGNormalization.MIN_MAX, per_channel = False, ** kwargs):
    """ Normalizes `eeg` (shape `(n_channels, n_samples)`) according to the normalization schema given by `normalize` """
    if isinstance(eeg, tf.Tensor):
        abs_fn, min_fn, max_fn, mean_fn, std_fn, divide_no_nan = (
            tf.abs, tf.reduce_min, tf.reduce_max, tf.reduce_mean, tf.math.reduce_std, tf.math.divide_no_nan
        )
    else:
        abs_fn, min_fn, max_fn, mean_fn, std_fn, divide_no_nan = (
            np.abs, np.min, np.max, np.mean, np.std, lambda a, b: np.divide(a, b, where = b != 0)
        )

    fn_kwargs = {} if not per_channel else {'axis' : -1, 'keepdims' : True}
    
    if normalize == EEGNormalization.NORMAL:
        eeg = divide_no_nan(eeg - mean_fn(eeg, ** fn_kwargs), std_fn(eeg, ** fn_kwargs))
    elif normalize == EEGNormalization.MAX:
        eeg = divide_no_nan(eeg, max_fn(abs_fn(eeg), ** fn_kwargs))
    elif normalize == EEGNormalization.MIN_MAX:
        eeg = eeg - min_fn(eeg, ** fn_kwargs)
        eeg = divide_no_nan(eeg, max_fn(eeg, ** fn_kwargs))
    elif normalize == EEGNormalization.TANH:
        eeg = eeg - min_fn(eeg, ** fn_kwargs)
        eeg = divide_no_nan(eeg, max_fn(eeg, ** fn_kwargs)) * 2. - 1.
    elif normalize == EEGNormalization.GlobalNormal:
        assert 'mean' in kwargs and 'std' in kwargs
        eeg = divide_no_nan(eeg - kwargs['mean'], kwargs['std'])
    elif normalize == EEGNormalization.GlobalMinMax:
        assert 'min' in kwargs and 'max' in kwargs
        eeg = divide_no_nan(eeg - kwargs['min'], kwargs['max'])

    return eeg

def normalize_dataset(* subsets, test = None):
    kwargs = {'axis' : 0, 'keepdims' : True}
    
    dataset = pd.concat(subsets) if len(subsets) > 1 else subsets[0]
    eeg = np.array(list(dataset['eeg'].values))
    test_eeg = np.array(list(test['eeg'].values)) if test is not None else None
    for channel in range(eeg.shape[1]):
        mean, std = np.mean(eeg[:, channel, :], ** kwargs), np.std(eeg[:, channel, :], ** kwargs)
        eeg[:, channel, :] = (eeg[:, channel, :] - mean) / std
        if test_eeg is not None:
            test_eeg[:, channel, :] = (test_eeg[:, channel, :] - mean) / std

    if test is not None: test['eeg'] = list(test_eeg)
    
    idx = 0
    for subset in subsets:
        subset['eeg'] = list(eeg[idx : idx + len(subset)])
        idx += len(subset)

    result = subsets
    if test is not None: result = result + (test, )
    
    return result if len(result) > 1 else result[0]

def compute_eeg_statistics(eeg, normalize, ** kwargs):
    if normalize not in _global_normalization_scheme: return {}

    if isinstance(eeg, pd.DataFrame): eeg = np.concatenate(list(eeg['eeg'].values), axis = -1)
    
    stats     = {}
    fn_kwargs = {'axis' : -1, 'keepdims' : True}
    if normalize == EEGNormalization.GlobalNormal:
        stats = {'mean' : np.mean(eeg, ** fn_kwargs), 'std' : np.std(eeg, ** fn_kwargs)}
    
    if normalize == EEGNormalization.GlobalMinMax:
        stats = {
            'min' : np.min(eeg, ** fn_kwargs),
            'max' : np.max(eeg, ** fn_kwargs) - np.min(eeg, ** fn_kwargs)
        }
    
    return tf.nest.map_structure(lambda s: tf.cast(s, tf.float32), stats)

def build_mne_struct(eeg, rate = None, channels = None, verbose = False, ** kwargs):
    import mne
    
    if isinstance(eeg, mne.io.BaseRaw): return eeg

    if not isinstance(eeg,dict):
        assert rate, 'You must specify the eeg sampling rate !'
        eeg = {'eeg' : eeg, 'rate' : rate, 'channels' : channels if channels else len(eeg)}
    
    if rate and 'rate' not in eeg: eeg['rate'] = rate
    if 'channels' not in eeg: eeg['channels'] = channels if channels else len(eeg)
    assert 'rate' in eeg, 'You must specify the eeg sampling rate !'
    
    n_channels = eeg['channels'] if isinstance(eeg['channels'], int) else len(eeg['channels'])
    infos  = mne.create_info(
        eeg['channels'], sfreq = eeg['rate'], ch_types = eeg.get('ch_types', ['eeg'] * n_channels)
    )
    
    return mne.io.RawArray(
        eeg['eeg'] if not hasattr(eeg['eeg'], 'numpy') else eeg['eeg'].numpy(), infos, verbose = verbose
    )

def _to_pytype(x):
    if isinstance(x, (int, float, str, bool)): return x
    if isinstance(x, tf.Tensor): x = x.numpy()
    return x.item()

def _create_df_iterator(df):
    for _, row in df.iterrows():
        yield row
    