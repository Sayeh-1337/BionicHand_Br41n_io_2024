# Copyright (C) 2023 Langlois Quentin, ICTEAM, UCLouvain. All rights reserved.
# Licenced under the GPL v3+ Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" section of the "README" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import itertools
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime, timezone
from scipy.io import loadmat

from loggers import timer, time_logger
from utils.eeg_utils import eeg_processing
from utils.generic_utils import convert_to_str
from utils.tensorflow_utils import execute_eagerly
from utils.wrapper_utils import dispatch_wrapper, partial

logger = logging.getLogger(__name__)

try:
    import mne
    mne_available = True
except ImportError:
    logger.warning('`mne` is not available, multiple features will therefore be removed !')
    mne_available = False

_device_rates = {'IN' : 128, 'MU' : 220, 'EP' : 128, 'MW' : 512}

_default_subject_infos = {'age' : 'age', 'sex' : ('sex', 'gender'), 'meas_date' : ('date', 'meas_date')}

_sleep_stages = ('W', 'N1', 'N2', 'N3', 'R')

_load_fn    = {}
_infos_fn   = {}
_events_fn  = {}
_write_fn   = {}

def load_eeg(data, rate, channels, ** kwargs):
    if isinstance(data, (dict, pd.Series)):
        eeg, _rate, _channels = data['eeg' if 'eeg' in data else 'filename'], data['rate'], data['channels']
        if 'start' in data: kwargs['start'] = data['start']
        if 'end' in data:   kwargs['end']   = data['end']
        if 'time' in data:  kwargs['time']  = data['time']
    else:
        eeg, _rate, _channels = data, kwargs.pop('rate', 0), kwargs.pop('channels', [])

    if channels is None: channels = tf.cast([], tf.string)
    return tf.transpose(read_eeg(
        eeg, rate = _rate, channels = _channels, target_rate = rate, target_channels = channels, ** kwargs
    )[0])

@dispatch_wrapper(_load_fn, 'File extension')
@execute_eagerly(signature = [
    eeg_processing.eeg_signature, eeg_processing.channels_signature, eeg_processing.rate_signature
], numpy = True)
def read_eeg(filename,
             target_rate     = None,
             target_channels = None,
             # processing config
             normalize    = False,
             detrend          = False,
             reduce_noise     = False,
             filtering_freqs  = None,
             remove_environmental_noise = False,

             start    = -1,
             end      = -1,
             time     = -1,

             rate     = None,
             channels = None,
             read_method  = None,
             
             dtype    = np.float32,
             
             ** kwargs
            ):
    """
        Generic method for EEG loading : internally calls the loading function associated to the filename extension, then it applies the expected processing
        
        Arguments :
            - filename  : the audio filename or raw audio (if raw, `rate` must be provided)
            - target_rate     : the rate to resample to (if required) (resampled with `scipy.signal`)
            - target_channels : the expected channels to keep / re-order
            
            - offset    : the number of samples to skip at the start / end of the audio
            - normalize : whether to normalize or not the audio (in range [0., 1.])
                - if a `float` is provided, divides by the value
            
            - start / end / time    : the time information to keep

            - rate     : the current EEG rate (only required if `filename` is the raw EEG)
            - channels : the current EEG channels (only required if `filename` is the raw EEG)
            - read_method   : string or callable, specific loading function to use
            
            - kwargs    : forwarded to the loading function, `reduce_noise` and `trim_silence`
        Returns : (eeg, channels, rate)
            - eeg      : the raw 2-D `np.ndarray` eeg data with shape `(n_channels, n_samples)`
            - channels : the EEG channels (equals to `target_channels` if provided)
            - rate     : the EEG rate (equals to `target_rate` if provided)
    """
    if isinstance(filename, (np.ndarray, tf.Tensor, list, tuple, bytes)):
        filename = convert_to_str(filename)
    
    if isinstance(filename, str):
        if read_method is None:
            read_method = _get_extension_method(filename, _load_fn)
        elif isinstance(read_method, str):
            if read_method not in _load_fn:
                raise ValueError('Unsupported EEG reading method !\n  Accepted : {}\n  Got : {}'.format(
                    tuple(_load_fn.keys()), read_method
                ))

            read_method = _load_fn[read_method]

        eeg, _channels, _rate = read_method(filename, rate = target_rate, channels = target_channels)
        if not channels: channels = _channels
        if not rate:     rate = _rate 
    elif mne_available and isinstance(filename, mne.io.BaseRaw):
        if channels is None: channels = filename.ch_names
        eeg, rate = filename, filename.info['sfreq']
    else:
        eeg = filename

    if hasattr(channels, 'shape') and len(channels.shape) == 0: channels = int(channels)
    else: channels = convert_to_str(channels)
    if target_channels is not None: target_channels = convert_to_str(target_channels)
    
    #logger.info('{} {} {} {} {} {} {}'.format(rate, target_rate, start, end, time, channels, target_channels))

    rate = int(rate)
    if target_rate is not None: target_rate = int(target_rate)
    
    assert channels is not None, 'You must provide the `channels` when passing raw EEG !'
    assert rate, 'You must provide the `rate` when passing raw EEG !'
    
    if isinstance(filename, str):
        eeg = eeg_processing.get_eeg_samples(eeg, rate, channels, start = start, end = end, time = time, ** kwargs)
    
    if mne_available and isinstance(eeg, mne.io.BaseRaw): eeg = eeg.get_data()

    if eeg.shape[1] == 0:
        logger.warning("EEG (shape {}) {} is empty !".format(eeg.shape, filename))
        return np.zeros((len(channels), rate), dtype = dtype), channels, rate

    eeg = eeg_processing.convert_eeg_dtype(eeg, dtype = dtype)
    
    if target_channels and channels != target_channels:
        eeg, channels = eeg_processing.rearrange_channels(eeg, channels = channels, target_channels = target_channels, ** kwargs)

    if normalize is not False:
        eeg = eeg_processing.normalize_eeg(eeg, normalize, ** kwargs)

    if target_rate is not None and target_rate > 0 and rate > 0 and target_rate != rate:
        eeg  = eeg_processing.resample_eeg(eeg, rate = rate, target_rate = target_rate, channels = channels, ** kwargs)
        rate = target_rate

    if detrend:
        eeg = eeg_processing.detrend_eeg(eeg, rate = rate, channels = channels, ** kwargs)

    if filtering_freqs is not None:
        eeg = eeg_processing.filter_eeg(eeg, rate = rate, channels = channels, freqs = filtering_freqs, ** kwargs)
        
    if remove_environmental_noise:
        eeg = eeg_processing.remove_environmental_noise(eeg, rate = rate, channels = channels, ** kwargs)
    
    if reduce_noise:
        eeg = eeg_processing.reduce_eeg_noise(eeg, rate = rate, channels = channels, ** kwargs)

    return eeg, channels, np.array(rate, dtype = np.int32)

def _read_eeg_mne(filename, io_method, ** kwargs):
    import mne
    data = getattr(mne.io, io_method)(filename, verbose = False) if isinstance(filename, str) else filename
    return data, data.ch_names, data.info['sfreq']

if mne_available:
    read_eeg.dispatch(partial(_read_eeg_mne, io_method = 'read_raw'), ('fif', 'fif.gz'))
    read_eeg.dispatch(partial(_read_eeg_mne, io_method = 'read_raw_edf'), 'edf')
    read_eeg.dispatch(partial(_read_eeg_mne, io_method = 'read_raw_gdf'), 'gdf')
    read_eeg.dispatch(partial(_read_eeg_mne, io_method = 'read_raw_brainvision'), 'eeg')
    read_eeg.dispatch(partial(_read_eeg_mne, io_method = 'read_raw_eeglab'), 'set')

@read_eeg.dispatch
def read_mat(filename, var_name = None, ** kwargs):
    data = filename
    if isinstance(filename, str):
        data = loadmat(filename, squeeze_me = True)
        if var_name is None: var_name = [k for k in data.keys() if not k.startswith('_')][0]
        data = data[var_name]
    
    eeg, channels, rate = [], kwargs.get('channels', []), kwargs.get('rate', None)
    for d in data if not np.issubdtype(data.dtype, np.void) else [data]:
        eeg.append(_get_matlab_field(d, ('x', 'data', 'eeg')))
        if not rate:     rate     = _get_matlab_field(d, ('rate', 'freq', 'fs', 'sampFreq'), required = True)
        if not channels: channels = _get_matlab_field(d, ('channels', 'chnames'), required = False)
    
    if any(eeg_i.shape[0] > eeg_i.shape[1] for eeg_i in eeg):
        eeg = [eeg_i.T for eeg_i in eeg]
    
    if not channels: channels = ['EEG-{}'.format(i) for i in range(1, len(eeg[0]) + 1)]
    return np.concatenate(eeg, axis = -1) if len(eeg) > 1 else eeg[0], channels, rate

@read_eeg.dispatch('hea')
def read_wfdb(filename, dtype = None, ** kwargs):
    import wfdb
    data = wfdb.rdrecord(filename.rstrip('.hea'))
    if dtype is None: dtype = np.int32 if 'uV' in data.units else np.float32
    return data.p_signal.T.astype(dtype), data.sig_name, data.fs

@read_eeg.dispatch
def read_npz(filename, ** kwargs):
    with np.load(filename) as file:
        return file['eeg'], convert_to_str(file['channels']), file['rate']


@dispatch_wrapper(_events_fn, 'File extension')
def read_eeg_events(filename,
                    add_signal  = False,
                    add_infos   = False,
                    event_names = None,
                    skip_empty  = True,
                    
                    start_label = None,
                    stop_label  = None,
                    
                    ** kwargs
                   ):
    """
        Generic method for EEG loading with events : internally calls the loading function associated to the filename extension
        
        Arguments :
            - filename    : the audio filename or raw audio (if raw, `rate` must be provided)
            - add_signal  : whether to add raw EEG data in the events information or not
            - event_names : mapping (`dict`) between event_id (keys) and event name (value)
            
            - kwargs    : forwarded to the loading function
                          if `add_signal`, these kwargs are also forwarded to `read_eeg`, which is called in the loading functions !
                          This enables processing on each individual event (e.g., noise reduction, resampling, ...)
        Returns :
            - events   : `list` of events information (`dict`) containing the following information (keys)
                - event_id : the id for the event label
                - start / end / time : event timing information
                - event_name : if `event_names` is provided, this key is equivalent to `event_names[event_id]`
                If `add_signal is True` these keys are also added:
                - eeg      : the raw 2-D `np.ndarray` eeg data with shape `(n_channels, n_event_samples)`
                - channels : the EEG channels (equals to `target_channels` if provided)
                - rate     : the EEG rate (equals to `target_rate` if provided)
    """
    if os.path.isdir(filename): filename = [os.path.join(filename, file) for file in os.listdir(filename)]
    if isinstance(filename, (list, tuple)):
        events, total_length = [], 0
        for file in os.listdir(filename):
            try:
                file_events = read_eeg_events(
                    file, add_siangl = add_signal, add_infos = add_infos, start_label = start_label, stop_label = stop_label, ** kwargs
                )
                if total_length:
                    for ev in file_events: ev.update({'start' : ev['start'] + total_length, 'end' : ev['end'] + total_length})
                total_length = file_events[-1]['end']
                events.extend(file_events)
            except RuntimeError:
                pass
        return events
    
    events = _get_extension_method(filename, _events_fn)(
        filename, add_signal = add_signal, add_infos = add_infos, event_names = event_names, ** kwargs
    )
    
    if start_label is not None or stop_label is not None:
        idx_start, idx_end = 0, None
        for i, ev in enumerate(events):
            if ev['event_id'] == start_label and not start_label: idx_start = i + 1
            if ev['event_id'] == stop_label:  idx_end = i
        events = events[idx_start : idx_end]
    
    if skip_empty:
        events = [ev for ev in events if ev['time'] > 0]
    
    if event_names:
        unknown, names = set(), set(event_names.values())
        for event in events:
            if event['event_id'] in event_names:
                event['event_name'] = event_names[event['event_id']]
            elif event.get('event_name', None) in event_names:
                event['event_name'] = event_names[event['event_name']]
            elif event.get('event_name', None) not in names:
                unknown.add(event['event_id'])
        
        if unknown:
            logger.warning('Unknown events detected !\n  Mapping : {}\n  Unknown : {}'.format(event_names, unknown))

    for event in events: event['rate'] = int(event['rate'])
    
    return events

def _read_events_mne(filename, add_signal, add_infos, io_method, event_names = None, keep_full_trial = True, ** kwargs):
    import mne
    
    data   = getattr(mne.io, io_method)(filename, preload = add_signal, verbose = False) if isinstance(filename, str) else filename
    annots = data.annotations
    subj_infos = {} if not add_infos else _get_infos_mne(data)
    
    eeg, channels, rate = (None, None, None) if not add_signal else read_eeg(data, ** kwargs)
    if not rate: rate = data.info['sfreq']
    
    starts    = np.around(annots.onset * rate).astype(np.int32)
    durations = np.around(annots.duration * rate).astype(np.int32)
    durations[durations <= 1] = 0
    description = annots.description
    if event_names: description = [event_names.get(d, d) for d in description]
    artifacts   = [False] * len(description)

    if keep_full_trial and 'start trial' in description:
        trial_starts, trial_duration, trial_des, trial_artifact = [], [], [], []
        trial_start, artifact, event = -1, False, None
        for start, duration, des in zip(starts, durations, description):
            if event:
                trial_starts.append(trial_start)
                trial_duration.append(start - trial_start)
                trial_des.append(event)
                trial_artifact.append(artifact)
                trial_start, artifact, event = -1, False, None
            
            if des == 'start trial':
                assert trial_start == -1, 'An event started while the previous has no event yet !'
                trial_start = start
            elif trial_start != -1:
                if des == 'rejected':
                    artifact = True
                else:
                    event = des
            else:
                trial_start, event = start, des
        
        if event:
            length = eeg.shape[1] if eeg is not None else len(data)
            trial_starts.append(trial_start)
            trial_duration.append(length - trial_start)
            trial_des.append(event)
            trial_artifact.append(artifact)

        starts, durations, description = trial_starts, trial_duration, trial_des
    
    infos = []
    for i, (start, duration, event_des) in enumerate(zip(starts, durations, description)):
        infos.append({
            'event_id' : event_des, 'start' : start, 'end' : start + duration, 'time' : duration, 'event_name' : event_des, ** subj_infos
        })
        if add_signal:
            infos[-1].update({
                'rate' : rate, 'channels' : channels, 'eeg' : eeg[:, start : start + duration]
            })
    return infos

if mne_available:
    read_eeg_events.dispatch(partial(_read_events_mne, io_method = 'read_raw'), ('fif', 'fif.gz'))
    read_eeg_events.dispatch(partial(_read_events_mne, io_method = 'read_raw_edf'), 'edf')
    read_eeg_events.dispatch(partial(_read_events_mne, io_method = 'read_raw_gdf'), 'gdf')
    read_eeg_events.dispatch(partial(_read_events_mne, io_method = 'read_raw_brainvision'), 'eeg')
    read_eeg_events.dispatch(partial(_read_events_mne, io_method = 'read_raw_eeglab'), 'set')

@read_eeg_events.dispatch
def read_events_mat(filename, add_signal, add_infos, var_name = None, ** kwargs):
    data = filename
    if isinstance(filename, str):
        data = loadmat(filename, squeeze_me = True)
        if var_name is None: var_name = [k for k in data.keys() if not k.startswith('_')][0]
        data = data[var_name]
    subj_infos = {} if not add_infos else get_infos_mat(data)
    
    channels, original_rate = kwargs.pop('channels', None), kwargs.pop('rate', None)
    
    infos, length = [], 0
    for d in data if not np.issubdtype(data.dtype, np.void) else [data]:
        eeg       = _get_matlab_field(d, ('x', 'data', 'eeg')) / 1e6
        starts    = _get_matlab_field(d, ('start', 'trial', 'event'), required = False)
        labels    = _get_matlab_field(d, ('y', 'label', 'marker'))
        artifacts = _get_matlab_field(d, ('artifacts', ), required = False)
        classes   = _get_matlab_field(d, ('classes', ), required = False)
        min_label = -1 if classes is None or len(labels) == 0 else np.min(labels)
        if eeg.shape[0] > eeg.shape[1]: eeg = eeg.T
        
        if starts is None:
            start, starts, _labels = 0, [], []
            for label, group in itertools.groupby(labels):
                length = len(list(group))
                starts.append(start)
                _labels.append(label)
                start += length
            starts, labels = np.array(starts), np.array(_labels)
        
        if add_signal:
            if not channels:
                channels = _get_matlab_field(d, ('channels', 'chnames'), required = False)
                if channels is None: channels = ['EEG-{}'.format(i+1) for i in range(len(eeg))]
            if not original_rate: original_rate = _get_matlab_field(d, ('rate', 'freq', 'fs', 'sampFreq'))
            eeg, channels, rate = read_eeg(eeg, channels = channels, rate = original_rate, ** kwargs)
            
            if 'target_rate' in kwargs and kwargs['target_rate'] != original_rate:
                starts = np.around(starts / original_rate * kwargs['target_rate']).astype(np.int32)
        
        if len(starts) > 0 and starts[0] > 0:
            starts, eeg = starts - starts[0], eeg[:, starts[0] :]

        for i, (start, label) in enumerate(zip(starts, labels)):
            end = starts[i + 1] if i < len(starts) - 1 else max(eeg.shape)
            infos.append({
                'event_id' : label, 'start' : start + length, 'end' : end + length, 'time' : end - start, ** subj_infos
            })
            if classes is not None:   infos[-1]['event_name'] = classes[label - min_label]
            if artifacts is not None: infos[-1]['artifact']   = artifacts[i]
            if add_signal:
                infos[-1].update({
                    'rate' : rate, 'channels' : channels, 'eeg' : eeg[:, start : end]
                })
        
        if len(starts) == 0:
            infos.append({
                'event_id' : -1, 'start' : length, 'end' : length + eeg.shape[1], 'time' : eeg.shape[1], ** subj_infos
            })
            if add_signal:
                infos[-1].update({
                    'rate' : rate, 'channels' : channels, 'eeg' : eeg
                })

        length += eeg.shape[1]
    
    return infos

@read_eeg_events.dispatch
def read_events_txt(filename, add_signal, add_infos, var_name = 'data', tqdm = lambda x: x, ** kwargs):
    lines = filename
    if isinstance(filename, str):
        with open(filename, 'r', encoding = 'utf-8') as file:
            lines = file.read().split('\n')
    
    events = collections.OrderedDict()
    for l in tqdm(lines):
        if not l: continue

        data_id, event_id, device, pos, label, size, eeg = l.split('\t')
        size, eeg = int(size), [float(v) / 1e6 for v in eeg.split(',')]

        if event_id not in events:
            events[event_id] = {
                'event_id'   : event_id,
                'event_name' : label,
                'device'     : device,
                'rate'       : _device_rates.get(device, -1),
                'channels'   : [],
                'eeg'        : []
            }
        events[event_id]['eeg'].append(eeg)
        events[event_id]['channels'].append(pos)
    
    events = list(events.values())
    total_length = 0
    for event in events:
        eeg, channels, rate = read_eeg(
            np.array(event['eeg'], dtype = np.float32), channels = event['channels'], rate = event['rate'], ** kwargs
        )
        event.update({
            'eeg'   : eeg,
            'rate'  : rate,
            'channels' : channels,
            'start' : total_length,
            'end'   : total_length + len(event['eeg'][0]),
            'time'  : len(event['eeg'][0])
        })
        total_length += eeg.shape[1]
    
    return events


@read_eeg_events.dispatch
def read_events_npy(filename, add_signal, add_infos, rate = None, channels = None, ** kwargs):
    assert rate, 'You must provie the `rate` argument'
    data = np.load(filename) if isinstance(filename, str) else filename
    
    eeg, labels = data[:, :-1].T, data[:, -1]
    if channels is None: channels = ['EEG-{}'.format(i + 1) for i in range(len(eeg))]

    original_rate       = rate
    eeg, channels, rate = read_eeg(eeg, rate = rate, channels = channels, ** kwargs)
    
    infos, start = [], 0
    for label, group in itertools.groupby(labels):
        length = len(list(group))
        if original_rate != rate: lenth = min(round(length / original_rate * rate), eeg.shape[1] - start)
        infos.append({
            'event_id' : label,
            'start'    : start,
            'end'      : start + length,
            'time'     : length,
            'rate'     : rate,
            'channels' : channels,
            'eeg'      : eeg[:, start : start + length]
        })
        start += length
    
    return infos


@read_eeg_events.dispatch('arousal')
def read_events_wfdb(filename, add_signal, add_infos, exclusive = _sleep_stages, ** kwargs):
    import wfdb
    hea_filename = filename.replace('.arousal', '.hea')
    
    annots  = wfdb.rdann(filename.rstrip('.arousal'), 'arousal')
    samples = np.array(annots.sample) - 1

    eeg, channels, rate, original_rate = None, None, None, None
    if add_signal:
        with time_logger.timer('read wfdb eeg'):
            eeg, channels, rate = read_eeg(hea_filename, ** kwargs)
        with open(hea_filename, 'r') as file:
            original_rate = int(file.read().split('\n')[0].split()[2])
        
        if rate != original_rate:
            samples = np.around(samples / original_rate * rate)

    infos, starts = [], {}
    for label, sample in zip(annots.aux_note, samples):
        end_label = label[:-1] if label[-1] == ')' else None
        if label in exclusive:
            for l in starts.keys():
                if l in exclusive:
                    end_label = l
                    break
        
        if end_label:
            if end_label not in starts:
                raise RuntimeError('{} ends without start !\n  Current starts : {}'.format(end_label, starts))
            
            start = starts.pop(end_label)
            infos.append({
                'event_id'   : end_label,
                'event_name' : end_label,
                'start'      : start,
                'end'        : sample,
                'time'       : sample - start
            })
            if add_signal:
                infos[-1].update({
                    'rate' : rate, 'channels' : channels, 'eeg' : eeg[:, start : sample]
                })
        if label.startswith('(') or label in exclusive:
            starts[label.lstrip('(')] = sample
    
    if starts:
        if eeg is None:
            with open(hea_filename, 'r') as file:
                length = int(file.read().split('\n')[0].split()[-1])
        else:
            length = eeg.shape[1]
        for label, start in starts.items():
            infos.append({
                'event_id'   : label,
                'event_name' : label,
                'start'      : start,
                'end'        : length,
                'time'       : length - start
            })
            if add_signal:
                infos[-1].update({
                    'rate' : rate, 'channels' : channels, 'eeg' : eeg[:, start : ]
                })
    return infos

@read_eeg_events.dispatch('arousal.mat')
def read_events_hdf5(filename, add_signal, add_infos, ** kwargs):
    def get_events(f, name):
        events = []
        if hasattr(f, 'keys'):
            if '#' in f: return events
            for k in f.keys():
                events.extend(get_events(f[k], k))
            return events
        
        data  = np.squeeze(np.array(f))
        index, max_index = 0, len(data) if not add_signal else eeg.shape[1]
        for v, n in itertools.groupby(data):
            length = len(list(n))
            if rate != original_rate:
                length = round(length / original_rate * rate)

            if v != 0:
                if index + length >= max_index - 1: length += 1
                events.append({
                    'event_id'   : v,
                    'event_name' : name,
                    'start'      : index,
                    'end'        : index + length,
                    'time'       : length
                })
                if add_signal:
                    infos[-1].update({
                        'rate' : rate, 'channels' : channels, 'eeg' : eeg[:, index : index + length]
                    })
            index += length
        return events
    
    import h5py
    
    eeg, channels, rate, original_rate = None, None, None, None
    if add_signal:
        eeg, channels, rate = read_eeg(hea_filename, ** kwargs)
        with open(hea_filename, 'r') as file:
            original_rate = int(file.read().split('\n').split()[2])
    
    with h5py.File(filename, 'r') as file:
        return get_events(file, None)

@read_eeg_events.dispatch
def _read_events_npz(filename, add_signal, add_infos, ** kwargs):
    with time_logger.timer('read_annotations'):
        with np.load(filename) as file:
            rate, starts, durations, description = file['rate'], file['starts'], file['durations'], convert_to_str(file['description'])
    
    with time_logger.timer('read_eeg'):
        eeg, channels, rate = (None, None, None) if not add_signal else read_eeg(filename, ** kwargs)
    with time_logger.timer('read_infos'):
        subj_infos = {} if not add_infos else get_infos_npz(filename)
    
    infos = []
    for i, (start, duration, event_des) in enumerate(zip(starts, durations, description)):
        infos.append({
            'event_id' : event_des, 'start' : start, 'end' : start + duration, 'time' : duration, 'event_name' : event_des, ** subj_infos
        })
        if add_signal:
            infos[-1].update({
                'rate' : rate, 'channels' : channels, 'eeg' : eeg[:, start : start + duration]
            })
    
    return infos

@read_eeg_events.dispatch('event.npy')
def _read_events_npy(filename, ** kwargs):
    return np.load(filename, allow_pickle = True)


@dispatch_wrapper(_infos_fn, 'File extension')
def get_subject_infos(filename, infos = _default_subject_infos):
    return _get_extension_method(filename, _infos_fn)(filename, keys = infos)

def _get_infos_mne(filename, keys = _default_subject_infos, io_method = 'read_raw'):
    import mne
    
    data  = getattr(mne.io, io_method)(filename, verbose = False) if isinstance(filename, str) else filename
    infos = {}
    if data.info['subject_info'] is not None:
        infos.update({
            k : v for k, v in data.info['subject_info'].items() if v is not None
        })
    
    if len(data._raw_extras) == 1 and 'subject_info' in data._raw_extras[0]:
        infos.update({
            k : v for k, v in data._raw_extras[0]['subject_info'].items() if v is not None
        })
        
    if not infos:
        logger.warning('The file {} does not have any subject information !'.format(filename))

    metadata = {k : None for k in keys.keys()}
    for k, group in keys.items():
        for k_i in group if isinstance(group, (list, tuple)) else [group]:
            if k_i in infos:
                metadata[k] = infos[k_i]
                break
            elif k_i == 'meas_date':
                metadata[k] = data.info[k_i]
    
    if isinstance(metadata.get('sex', None), int):
        if metadata['sex'] == 1: metadata['sex'] = 'Male'
        elif metadata['sex'] == 2: metadata['sex'] = 'Female'
        else: metadata['sex'] = 'Unknown'
    
    return metadata

if mne_available:
    get_subject_infos.dispatch(partial(_get_infos_mne, io_method = 'read_raw'), ('fif', 'fif.gz'))
    get_subject_infos.dispatch(partial(_get_infos_mne, io_method = 'read_raw_edf'), 'edf')
    get_subject_infos.dispatch(partial(_get_infos_mne, io_method = 'read_raw_gdf'), 'gdf')
    get_subject_infos.dispatch(partial(_get_infos_mne, io_method = 'read_raw_brainvision'), 'eeg')
    get_subject_infos.dispatch(partial(_get_infos_mne, io_method = 'read_raw_eeglab'), 'set')

@get_subject_infos.dispatch
def get_infos_mat(filename, keys = _default_subject_infos, var_name = 'data', ** kwargs):
    data = loadmat(filename, squeeze_me = True)[var_name] if isinstance(filename, str) else filename
    if not np.issubdtype(data.dtype, np.void): data = data[0]
    return {k : _get_matlab_field(data, v, required = False) for k, v in keys.items()}

@get_subject_infos.dispatch
def get_infos_npz(filename, keys = _default_subject_infos, ** kwargs):
    infos = {}
    with np.load(filename) as file:
        _files = file.keys()
        for k, group in keys.items():
            for ki in group:
                if ki in _files:
                    infos[k] = file[ki].item()
                    break
    return infos


def _get_extension_method(filename, methods):
    if not isinstance(filename, str): raise ValueError('`filename` must be an EEG filename (str) but got {}'.format(filename))
    
    for ext in sorted(methods.keys(), reverse = True, key = len):
        if filename.endswith(ext): return methods[ext]

    raise RuntimeError('Unsupported EEG file extension !\n  Accepted : {}\n  Got : {}'.format(
        tuple(methods.keys()), filename
    ))

def _get_matlab_field(data, candidates, required = True):
    candidates = [c.lower() for c in candidates]
    for field in data.dtype.fields.keys():
        if field.lower() in candidates: return data[field].item()
    if required: raise RuntimeError('No candidates found !\n  Fields : {}\n  Candidates : {}'.format(data.dtype.fields.keys(), candidates))
    return None

def rewrite_eeg_file(filename, events = None, new_format = 'fif.gz', output_dir = None, ** kwargs):
    """ Rewrite `events` (or `read_eeg_events(filename, ** kwargs)`) to a new file format, and returns a tuple (filename, events) """
    basename, ext = os.path.splitext(filename)
    if 'target_rate' not in kwargs:
        new_filename  = '{}.{}'.format(basename, new_format)
    else:
        new_filename   = '{}-{}.{}'.format(basename, kwargs['target_rate'], new_format)
    
    if output_dir: new_filename = os.path.join(output_dir, os.path.basename(new_filename))
    
    if events is None:
        with time_logger.timer('read_events'):
            events = read_eeg_events(filename, add_signal = True, add_infos = True, ** kwargs)
    
    with time_logger.timer('write_eeg'):
        filename = write_eeg(new_filename, events)
    return filename, events

@dispatch_wrapper(_write_fn, 'File extension')
def write_eeg(filename, events, ** kwargs):
    if not events: raise RuntimeError('Empty events !')
    return _get_extension_method(filename, _write_fn)(filename, events, ** kwargs)

def _write_eeg_mne(filename, events, ** kwargs):
    import mne
    
    events   = sorted(events, key = lambda e: (e['start'], - e['time']))
    
    eeg      = []
    annots   = []
    rate     = events[0]['rate']
    channels = events[0]['channels']
    infos    = {k : v for k, v in events[0].items() if k in _default_subject_infos}
    if isinstance(infos.get('sex', None), str):
        if infos['sex'].lower() in ('m', 'h', 'male'): infos['sex'] = 1
        elif infos['sex'].lower() in ('f', 'female'):  infos['sex'] = 2
        else: infos['sex'] = 0
    
    last_end, skipped = 0, 0
    for event in events:
        if event['start'] > last_end: skipped += event['start'] -last_end
        if event['end'] > last_end:
            event_eeg = event['eeg']
            if event['start'] < last_end:
                event_eeg = event_eeg[:, last_end - event['start'] :]
            eeg.append(event_eeg)
            last_end = event['end']
        annots.append([event['start'] - skipped, event['time'], event['event_name']])
    
    eeg = np.concatenate(eeg, axis = -1)
    mne_struct = eeg_processing.build_mne_struct(
        eeg, rate = rate, channels = channels
    )
    mne_struct.set_meas_date(events[0].get('meas_date', kwargs.get('meas_date', datetime.now(timezone.utc))))
    
    annots = mne.Annotations(
        onset       = np.array([ann[0] for ann in annots]) / rate,
        duration    = np.array([ann[1] for ann in annots]) / rate,
        description = [ann[2] for ann in annots],
        orig_time   = mne_struct.info['meas_date']
    )

    mne_struct.set_annotations(annots)
    mne_struct.info['subject_info'] = infos
    if filename.endswith(('.fif', '.fif.gz')):
        mne_struct.save(filename, overwrite = True, verbose = False)
    else:
        mne_struct.export(filename, overwrite = True)

    return filename

if mne_available:
    write_eeg.dispatch(_write_eeg_mne, 'eeg')
    write_eeg.dispatch(_write_eeg_mne, 'edf')
    write_eeg.dispatch(_write_eeg_mne, 'fif')
    write_eeg.dispatch(_write_eeg_mne, 'fif.gz')

@write_eeg.dispatch('event.npy')
def _write_eeg_npz(filename, events, ** kwargs):
    np.save(filename, events)
    return filename

@write_eeg.dispatch
def _write_eeg_npz(filename, events, ** kwargs):
    events   = sorted(events, key = lambda e: (e['start'], - e['time']))
    
    rate, channels = events[0]['rate'], events[0]['channels']
    infos    = {k : v for k, v in events[0].items() if k in _default_subject_infos}
    
    eeg, starts, durations, descriptions = [], [], [], []

    last_end, skipped = 0, 0
    for event in events:
        if event['start'] > last_end: skipped += event['start'] -last_end
        if event['end'] > last_end:
            event_eeg = event['eeg']
            if event['start'] < last_end:
                event_eeg = event_eeg[:, last_end - event['start'] :]
            eeg.append(event_eeg)
            last_end = event['end']
        
        starts.append(event['start'] - skipped)
        durations.append(event['time'])
        descriptions.append(event['event_name'])
    
    eeg = (np.concatenate(eeg, axis = -1) * 1e6).astype(np.int16)
    
    if 'meas_date' in infos:
        date = infos.pop('meas_date')
        if isinstance(date, datetime):
            infos['meas_date'] = '{}-{}-{}'.format(date.year, date.month, date.day)
    
    np.savez(
        filename,
        eeg      = eeg,
        rate     = rate,
        channels = np.array(channels),
        starts   = np.array(starts, dtype = np.int32),
        durations   = np.array(durations, dtype = np.int32),
        description = np.array(descriptions),
        ** infos
    )
    return filename
