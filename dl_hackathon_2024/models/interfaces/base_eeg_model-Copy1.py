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

import os
import logging
import pandas as pd
import tensorflow as tf

from utils import get_enum_item
from utils.eeg_utils import EEGNormalizeMode, tf_resample_eeg, normalize_eeg, compute_eeg_statistics
from models.interfaces.base_audio_model import BaseAudioModel

logger = logging.getLogger(__name__)

DEFAULT_EEG_RATE    = 160

class BaseEEGModel(BaseAudioModel):
    def _init_eeg(self,
                  channels,
                  audio_rate    = DEFAULT_EEG_RATE,
                  keep_spatial_information  = True,
                  
                  mel_fn    = None,
                  csp_fn    = None,

                  normalize_mode   = EEGNormalizeMode.MIN_MAX,
                  normalize_config = None,
                  normalize_per_channel  = True,
                  
                  max_input_length = None,
                  use_fixed_length_input = False,

                  ** kwargs
                 ):
        if use_fixed_length_input:
            assert max_input_length, 'You must specify the fixed input length !'
        
        kwargs.setdefault('audio_format', 'raw' if mel_fn is None else 'mel')
        self._init_audio(
            audio_rate = audio_rate, mel_fn = mel_fn, ** kwargs
        )
        
        self.channels   = channels if not isinstance(channels, int) else [
            'Chan {}'.format(i+1) for i in range(channels)
        ]
        self.keep_spatial_information   = keep_spatial_information
        
        self.max_input_length = max_input_length
        self.use_fixed_length_input = use_fixed_length_input
        
        self.normalize_mode   = get_enum_item(normalize_mode, EEGNormalizeMode) if normalize_mode is not None else None
        self.normalize_per_channel  = normalize_per_channel
        self.normalize_config = {} if not normalize_config else normalize_config
        for k in ('min', 'max', 'mean', 'std'):
            if 'k' in self.normalize_config: self.normalize_config = tf.cast(self.normalize_config[k], tf.float32)
        
        self.csp_fn  = None
        if csp_fn is not None:
            from utils.eeg_utils.csp import CSP
            # Initialization of mel fn
            if isinstance(csp_fn, CSP):
                self.csp_fn    = csp_fn
            elif isinstance(csp_fn, str):
                self.csp_fn = CSP.load_from_file(csp_fn)
            else:
                if not isinstance(csp_fn, dict): csp_fn = {'n_csp' : csp_fn}
                if audio_rate: csp_fn['sampling_rate'] = audio_rate
                self.csp_fn    = CSP(** csp_fn)
    
    @property
    def optimizer(self):
        return self.get_optimizer()
    
    @property
    def use_csp(self):
        return self.csp_fn is not None
    
    @property
    def csp_file(self):
        return os.path.join(self.save_dir, 'csp.json')
    
    @property
    def n_eeg_channels(self):
        return len(self.channels)
    
    @property
    def n_csp_features(self):
        return self.csp_fn.n_features if self.use_csp else -1
    
    @property
    def eeg_signature(self):
        length = None if not self.use_fixed_length_input else self.max_input_length
        
        if self.use_mel_fn:
            if self.keep_spatial_information:
                shape = (None, length, self.n_eeg_channels, self.n_mel_channels)
            else:
                shape = (None, length, self.n_mel_channels, self.n_eeg_channels)
        elif self.use_csp:
            shape = (None, self.n_csp_features, 1)
        elif self.keep_spatial_information:
            shape = (None, length, self.n_eeg_channels, 1)
        else:
            shape = (None, length, self.n_eeg_channels)
        
        return tf.TensorSpec(shape = shape, dtype = tf.float32)
    
    def _str_eeg(self):
        des = self._str_audio()
        des += "- EEG channels ({}) : {}\n".format(self.n_eeg_channels, self.channels)
        if self.use_csp:
            des += "- # CSP features : {}\n".format(self.n_csp_features)
        else:
            des += "- Keep spatial information : {}\n".format(self.keep_spatial_information)
        return des
    
    def compute_normalization_config(self, dataset):
        self.normalize_config.update(tf.nest.map_structure(
            lambda v: tf.cast(v, tf.float32), compute_eeg_statistics(dataset, self.normalize_mode)
        ))
        
    def get_raw_eeg(self, data, ** kwargs):
        """
            Return raw EEG data from `data` after normalization
            
            Arguments :
                - data : dict / pd.Series / np.ndarray / tf.Tensor, the raw EEG data (if dict, should have a `eeg` key)
            Return :
                - eeg : raw EEG data with shape [eeg_samples, self.n_channels(, 1)] (if `keep_spatial_feature`)
        """
        eeg = tf.cast(data['eeg'] if isinstance(data, (dict, pd.Series)) else data, tf.float32)
        
        if len(tf.shape(eeg)) != 2:
            raise ValueError('Unsupported EEG shape : {}'.format(tf.shape(eeg)))
        
        if self.normalize_mode is not None:
            eeg = normalize_eeg(
                eeg,
                per_channel    = self.normalize_per_channel,
                normalize_mode = self.normalize_mode,
                ** self.normalize_config
            )

        eeg = tf.transpose(eeg)
        
        if self.keep_spatial_information and not self.use_mel_fn:
            eeg = tf.expand_dims(eeg, axis = -1)
        
        return eeg
    
    def get_mel_eeg(self, eeg, ** kwargs):
        # mel.shape == [n_eeg_channels, length, n_mel_channels]
        mel = self.mel_fn(eeg)
        
        if self.keep_spatial_information:
            mel = tf.transpose(mel, [1, 0, 2])
        else:
            mel = tf.transpose(mel, [1, 2, 0])
        
        return mel
    
    def get_csp_eeg(self, data, ** kwargs):
        assert isinstance(data, (dict, pd.Series)) and ('id' in data or 'subj_id' in data)
        
        csp_features = data['eeg']
        if tf.shape(csp_features)[0] == self.n_eeg_channels or tf.shape(csp_features)[1] != self.n_csp_features:
            subj_id     = data['subj_id'] if 'subj_id' in data else data['id']
            csp_features    = self.csp_fn(csp_features, subj_id = subj_id)
            csp_features    = tf.squeeze(csp_features, axis = 0),
        
        return tf.expand_dims(csp_features, axis = 1)
    
    def get_eeg(self, data, truncate = True, ** kwargs):
        if isinstance(data, list):
            return [self.get_eeg(data_i, truncate = truncate, ** kwargs) for data_i in data]
        elif isinstance(data, pd.DataFrame):
            return [self.get_eeg(row, truncate = truncate, ** kwargs) for idx, row in data.iterrows()]

        if isinstance(data, (dict, pd.Series)):
            data['eeg'] = tf.cast(data['eeg'], tf.float32)
            if 'rate' in data and data['rate'] != self.audio_rate:
                data['eeg'] = tf_resample_eeg(
                    data['eeg'], rate = data['rate'], target_rate = self.audio_rate
                )
        
        if self.use_csp:
            return self.get_csp_eeg(data, ** kwargs)
        
        eeg = self.get_raw_eeg(data, ** kwargs)
        
        if self.use_mel_fn:
            eeg = self.get_mel_eeg(eeg, ** kwargs)
        
        if self.max_input_length is not None and truncate and tf.shape(eeg)[0] > self.max_input_length:
            start = tf.random.uniform(
                (),
                minval = 0, 
                maxval = tf.shape(eeg)[0] - self.max_input_length,
                dtype  = tf.int32
            )
            eeg = eeg[start : start + self.max_input_length]
        
        return eeg
    
    def train(self, x, * args, train_csp_samples = None, ** kwargs):
        if self.use_csp:
            assert 'validation_data' in kwargs, 'When using CSP, you must pre-split the dataset to be sure that the CSP features are initialized based on the train_set and not on the valid_set !'
            self.csp_fn.fit_dataset(
                x if not hasattr(x, 'dataset') else x.dataset, n_sample = train_csp_samples
            )
        
        if self.normalize_mode in (EEGNormalizeMode.GlobalNormal, EEGNormalizeMode.GlobalMinMax) and not self.normalize_config:
            assert 'validation_data' in kwargs, 'When using global normalization scheme, you must pre-split the dataset to be sure that the normalization statistics are computed based on the train_set and not on the valid_set !'
            self.compute_normalization_config(x if not hasattr(x, 'dataset') else x.dataset)
        
        return super(BaseEEGModel, self).train(x, * args, ** kwargs)
    
    def get_config_eeg(self, * args, ** kwargs):
        if self.use_csp:
            self.csp_fn.save_to_file(self.csp_file)
        
        config = self.get_config_audio(* args, ** kwargs)
        config.update({
            'channels'  : self.channels,
            'csp_fn'    : self.csp_file if self.use_csp else None,
            'normalize_mode'    : self.normalize_mode,
            'normalize_config'  : self.normalize_config,
            'normalize_per_channel'  : self.normalize_per_channel,
            'max_input_length'       : self.max_input_length,
            'use_fixed_length_input' : self.use_fixed_length_input
        })
            
        return config
