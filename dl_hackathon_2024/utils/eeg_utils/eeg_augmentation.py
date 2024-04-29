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

import numpy as np
import pandas as pd

def split_channels(data):
    is_df = isinstance(data, pd.DataFrame)
    if is_df: data = data.to_dict('records')

    splitted = []
    for d in data:
        for i, c in enumerate(d['channels']):
            new_d = d.copy()
            new_d.update({'channels' : [c], 'n_channels' : 1})
            if 'eeg' in new_d: new_d['eeg'] = new_d['eeg'][i : i + 1]
            splitted.append(new_d)
    
    if is_df: splitted = pd.DataFrame(splitted)
    return splitted
    
def augment_dataset_windowing(data, window_len, window_step = -1, n_window = -1):
    assert window_step != -1 or n_window != -1, 'You must provide at least the number of windows `n_window` or the sliding step `window_step`'

    samples   = data.to_dict('records')
    augmented = []
    for idx, sample in enumerate(samples):
        rate   = sample['rate']
        length = sample['eeg'].shape[1] if 'eeg' in sample else sample['time']
        if isinstance(length, float): length = int(length * rate)

        win_samples  = window_len if isinstance(window_len, int) else int(window_len * rate)
        if window_step not in (-1, None):
            step_samples = window_step if isinstance(window_step, int) else int(window_step * rate)
            steps = list(range(0, length - win_samples, step_samples))
        else:
            steps = np.unique(np.linspace(0, length - win_samples, n_window).astype(np.int32))
        
        if n_window != -1:
            steps = steps[: n_window]
        
        for i, start in enumerate(steps):
            augmented.append({
                ** sample,
                'start' : sample['start'] + start,
                'end'   : sample['start'] + start + win_samples,
                'time'  : win_samples,
                'original_sample' : idx,
                'window_index'    : i
            })
            if 'eeg' in sample: augmented[-1]['eeg'] = sample['eeg'][:, start : start + win_samples]
    
    return pd.DataFrame(augmented)

