
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from utils.eeg_utils import load_eeg
from .grouper_generator import GrouperGenerator
from .siamese_generator import SiameseGenerator

class EEGLoader:
    @property
    def processed_signature(self):
        return {
            'id'       : tf.TensorSpec(shape = (), dtype = tf.string),
            'eeg'      : tf.TensorSpec(shape = (None, None), dtype = tf.float32),
            'rate'     : tf.TensorSpec(shape = (), dtype = tf.int32),
            'channels' : tf.TensorSpec(shape = (None, ), dtype = tf.string)
        }

    def seg_eeg_infos(self, dataset, kwargs = {}, ** _):
        kwargs.update({'cache_size' : 0, 'preload' : False, 'id_column' : 'label', 'file_column' : 'data_idx'})
        dataset['data_idx'] = list(range(len(dataset)))

    def load_file(self, data_idx):
        data = self.dataset.iloc[data_idx]
        return {k : data[k] for k in ('id', 'eeg', 'rate', 'channels')}

class EEGGrouperGenerator(EEGLoader, GrouperGenerator):
    def __init__(self, dataset, rate = -1, channels = None, ** kwargs):
        self.seg_eeg_infos(dataset, rate = rate, channels = channels, kwargs = kwargs)
        
        super().__init__(dataset, ** kwargs)

class EEGSiameseGenerator(EEGLoader, SiameseGenerator):
    def __init__(self, dataset, rate = -1, channels = None, ** kwargs):
        self.seg_eeg_infos(dataset, kwargs, rate = rate, channels = channels)
        
        super().__init__(dataset, ** kwargs)
