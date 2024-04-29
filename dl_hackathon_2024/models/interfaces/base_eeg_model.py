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
import numpy as np
import pandas as pd
import tensorflow as tf

from hparams import HParams
from utils.audio import MelSTFT
from loggers import timer, time_logger
from models.interfaces.base_model import BaseModel, compute_distributed_loss
from utils.eeg_utils import EEGNormalization, CSP, EuclidianAlignment, load_eeg, read_eeg, compute_eeg_statistics

logger = logging.getLogger(__name__)

DEFAULT_NORMALIZATION_CONFIG = {'normalize' : EEGNormalization.MIN_MAX}

EEGTrainingParams = HParams(max_input_length = None, mixup = False, mixup_prct = 0.75)

class BaseEEGModel(BaseModel):
    def infer(self, inputs, ids = None, channels = None, training = False, mask_by_id = False, ** kwargs):
        """
            Perform the model inference on the given data

            Arguments :
                - inputs : the processed model inputs
                - ids    : the subject id for the given inputs (i.e., `ids[i]` is the subject id for `inputs[i]`)
                - channels : the channels for the given inputs
                - mask_by_id : whether to mask labels that do not belong to the given subject
                               this may be useful for models trained with subject-specific labels
            Return :
                - labels     : list of the candidate labels
                - scores     : `tf.Tensor` with shape `(len(inputs), len(labels))`
                               representing, for each input, the score for each possible label
        """
        raise NotImplementedError()

    def get_output(self, data):
        """ Return the label(s) associated with the given data """
        raise NotImplementedError()
    
    def _init_eeg(self,
                  rate,
                  channels = None,
                  keep_spatial_information  = 'auto',
                  
                  max_input_length = None,
                  use_fixed_length_input = False,
                  
                  normalization_config = DEFAULT_NORMALIZATION_CONFIG,

                  use_ea    = False,
                  mel_fn    = None,
                  csp_fn    = None,

                  ** kwargs
                 ):
        if use_fixed_length_input:
            assert max_input_length, 'You must specify the fixed input length !'

        self.nom        = kwargs['nom']
        
        self.rate       = rate
        self.channels   = channels
        self._keep_spatial_information   = keep_spatial_information
        
        self.max_input_length = max_input_length if not isinstance(max_input_length, float) else int(max_input_length * rate)
        self.use_fixed_length_input = use_fixed_length_input

        self.use_ea = use_ea
        self.normalization_config = normalization_config
        
        self.mel_fn = None
        self.csp_fn = None
        self.ea_fn  = None if not use_ea else EuclidianAlignment.load_from_file(self.ea_file)
        
        if mel_fn is not None:
            if not isinstance(mel_fn, MelSTFT):
                if isinstance(mel_fn, dict):
                    mel_fn = MelSTFT.create(** mel_fn)
                else:
                    mel_fn = MelSTFT.load_from_file(mel_fn)
            
            self.mel_fn = mel_fn
        
        if csp_fn is not None:
            # Initialization of mel fn
            if not isinstance(csp_fn, CSP):
                if isinstance(csp_fn, str):
                    self.csp_fn = CSP.load_from_file(csp_fn)
                else:
                    if not isinstance(csp_fn, dict): csp_fn = {'n_csp' : csp_fn}
                    if rate: csp_fn['sampling_rate'] = rate
                    csp_fn    = CSP(** csp_fn)
            
            self.csp_fn    = csp_fn
    
    def _build_model(self, ** config):
        config.update({'input_shape' : self.eeg_shape[1:]})
        return super(BaseEEGModel, self)._build_model(model = config)

    @property
    def keep_spatial_information(self):
        if self.channels is None: return True
        if self._keep_spatial_information != 'auto':
            return self._keep_spatial_information
        elif self.use_mel or not hasattr(self, 'model'):
            return False
        return len(self.model.input_shape) == 4
    
    @property
    def optimizer(self):
        return self.get_optimizer()
    
    @property
    def use_mel(self):
        return self.mel_fn is not None

    @property
    def use_csp(self):
        return self.csp_fn is not None

    @property
    def ea_file(self):
        return os.path.join(self.save_dir, 'ea.json') if self.use_ea else None

    @property
    def mel_file(self):
        return os.path.join(self.save_dir, 'mel_config.json') if self.use_mel else None
    
    @property
    def csp_file(self):
        return os.path.join(self.save_dir, 'csp_config.json') if self.use_csp else None

    @property
    def n_eeg_channels(self):
        if isinstance(self.channels, list): return len(self.channels)
        return self.channels
    
    @property
    def n_mel_channels(self):
        return self.mel_fn.n_mel_channels if self.use_mel else -1
    
    @property
    def n_csp_features(self):
        return self.csp_fn.n_features if self.use_csp else -1

    @property
    def is_single_channel(self):
        return self.channels == 1
    
    @property
    def eeg_shape(self):
        if hasattr(self, 'model'): return self.model.input_shape
        length = None if not self.use_fixed_length_input else self.max_input_length
        
        if self.use_mel:
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
        
        return shape

    @property
    def eeg_signature(self):
        return tf.TensorSpec(shape = self.eeg_shape, dtype = tf.float32)
    
    @property
    def training_hparams_eeg(self):
        return EEGTrainingParams()

    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_eeg)
    
    def _str_eeg(self):
        des = "- EEG rate : {}\n".format(self.rate)
        if self.channels:
            des += "- EEG channels ({}) : {}\n".format(self.n_eeg_channels, self.channels)
        else:
            des += "- EEG channels : variable\n"

        if self.use_ea:
            des += "- Use Euclidian Alignment (EA) : True\n"
        
        if self.use_mel:
            des += "- # Mel channels : {}\n".format(self.n_mel_channels)
        elif self.use_csp:
            des += "- # CSP features : {}\n".format(self.n_csp_features)
        else:
            des += "- Keep spatial information : {}\n".format(self.keep_spatial_information)
        
        if self.normalization_config:
            des += "- Normalization config : {}\n".format(self.normalization_config)
        return des

    def infer_multi_channels(self, inputs, *, channels = None, ** kwargs):
        """
            Perform the model inference on the given multi-channels data, while the model has been trained on single-channel
            This method calls `self.infer` on each channel of the given inputs, and stacks the results

            Arguments :
                - inputs   : the processed multi-channels model inputs
                - channels : the channels for the given inputs
                - kwargs   : forwarded to `self.infer`
            Return :
                - labels     : list of the candidate labels
                - scores     : `tf.Tensor` with shape `(len(inputs), n_channels, len(labels))`
                               representing, for each input (1st axis), for each channel (2nd axis), the score for each possible label (3rd axis)
        """
        results = [
            self.infer(inputs[:, :, i : i + 1], channels = [chans[i] for chans in channels], ** kwargs)
            for i in range(inputs.shape[2])
        ]
        return results[0], np.stack([res[1] for res in results], axis = 1)

    def set_normalization_stats(self, stats, ** kwargs):
        if 'normalize' not in self.normalization_config: return
        
        if not isinstance(stats, dict): stats = compute_eeg_statistics(stats, ** self.normalization_config)
        for k, v in stats.items(): self.normalization_config.setdefault(k, v)
    
    def get_raw_eeg(self, data, ** kwargs):
        """
            Return raw EEG data from `data` after normalization
            
            Arguments :
                - data : dict / pd.Series / np.ndarray / tf.Tensor, the EEG data (see `load_eeg` for more information)
            Return :
                - eeg : raw EEG data with shape [eeg_samples, self.n_channels(, 1)] (if `keep_spatial_feature`)
        """
        if self.use_ea:
            assert isinstance(data, (dict, pd.Series))
            eeg = self.ea_fn(load_eeg(data, rate = -1, channels = self.channels), subject = data['id'])
            data = {'eeg' : tf.transpose(eeg), 'channels' : self.channels, 'rate' : data['rate']}

        eeg = load_eeg(
            data,
            rate     = self.rate,
            channels = self.channels if self.channels not in (None, 1) else [],
            ** self.normalization_config
        )

        if self.keep_spatial_information and not self.use_mel and not self.use_csp:
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
    
    def get_csp_eeg(self, eeg, data, ** kwargs):
        assert isinstance(data, (dict, pd.Series)) and ('id' in data or 'subj_id' in data)
        
        subj_id     = data['subj_id' if 'subj_id' in data else 'id']
        csp_features    = self.csp_fn(eeg, subj_id = subj_id)
        csp_features    = tf.squeeze(csp_features, axis = 0),
        
        return tf.expand_dims(csp_features, axis = 1)
    
    def get_eeg(self, data, truncate = True, ** kwargs):
        if isinstance(data, list):
            return [self.get_eeg(data_i, truncate = truncate, ** kwargs) for data_i in data]
        elif isinstance(data, pd.DataFrame):
            return [self.get_eeg(row, truncate = truncate, ** kwargs) for idx, row in data.iterrows()]

        eeg = self.get_raw_eeg(data, ** kwargs)

        if self.use_csp:
            return self.get_csp_eeg(eeg, data, ** kwargs)
        
        if self.use_mel:
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

    def encode_data(self, data):
        return self.get_input(data), self.get_output(data)

    def train_step(self, batch):
        inputs, target = batch

        if self.mixup:
            factor = tf.random.uniform((), minval = 0.5, maxval = 0.5)
            inputs = tf.cond(
                tf.random.uniform(()) < self.mixup_prct,
                lambda: factor * inputs + (1. - factor) * tf.random.shuffle(inputs),
                lambda: inputs
            )

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training = True)
            
            replica_loss = compute_distributed_loss(
                self.model_loss, target, y_pred,
                global_batch_size = self.global_batch_size,
                nb_loss = 1
            )

        grads = tape.gradient(replica_loss, self.model.trainable_variables)
        self.model_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.update_metrics(target, y_pred)
        return replica_loss

    def fit(self, x, * args, train_csp_samples = None, ** kwargs):
        if self.use_ea or self.use_csp or self.normalization_config:
            assert 'validation_data' in kwargs, 'When using global normalization (e.g., EA or CSP), `validation_data` must be provided !'
        
            if self.use_ea:
                self.ea_fn.fit(x if not hasattr(x, 'dataset') else x.dataset)
            
            if self.use_csp:
                self.csp_fn.fit_dataset(
                    x if not hasattr(x, 'dataset') else x.dataset, n_sample = train_csp_samples
                )
            
            if self.normalization_config:
                self.set_normalization_stats(x if not hasattr(x, 'dataset') else x.dataset)
        
        return super(BaseEEGModel, self).fit(x, * args, ** kwargs)

    def train(self, x, * args, train_csp_samples = None, ** kwargs):
        if self.use_ea or self.use_csp or self.normalization_config:
            assert 'validation_data' in kwargs, 'When using global normalization (e.g., EA or CSP), `validation_data` must be provided !'
        
            if self.use_ea:
                self.ea_fn.fit(x if not hasattr(x, 'dataset') else x.dataset)
            
            if self.use_csp:
                self.csp_fn.fit_dataset(
                    x if not hasattr(x, 'dataset') else x.dataset, n_sample = train_csp_samples
                )
            
            if self.normalization_config:
                self.set_normalization_stats(x if not hasattr(x, 'dataset') else x.dataset)
        
        return super(BaseEEGModel, self).train(x, * args, ** kwargs)

    def embed(self, data, batch_size = 256, model = None):
        if isinstance(data, pd.DataFrame): data = data.to_dict('records')
        if model is None: model = self
        embeddings = []
        for i in range(0, len(data), batch_size):
            inp = self.get_input(data[i : i + batch_size])
            inp = tf.stack(inp, axis = 0) if self.use_fixed_length_input else tf.cast(pad_batch(inp, dtype = np.float32), tf.float32)
            embeddings.append(model(inp, training = False))
        return tf.concat(embeddings, axis = 0)
            
    def build_samples(self, data, ** kwargs):
        assert isinstance(data, pd.DataFrame)
        return {
            'ids'      : data['id'].values,
            'channels' : np.squeeze(list(data['channels'].values)),
            'embeddings' : self.embed(data, ** kwargs),
            'labels'   : self.get_output(data['event' if 'event' in data else 'label'].values)
        }

    @timer
    def _compute_metrics(self, labels, scores):
        if isinstance(scores, tf.Tensor): scores = scores.numpy()
        is_binary = scores.shape[-1] == 1
        
        if not is_binary:
            preds = np.argmax(scores, axis = -1)
            probs = scores / np.sum(scores, axis = -1, keepdims = True)
        else:
            probs = scores
            preds = (scores > 0.5).astype(np.int32)

        results = []
        for pred, score, prob in zip(preds, scores, probs):
            results.append({
                'pred'    : labels[pred],
                'pred_id' : pred,
                'score'   : score[pred] if not is_binary else score,
                'prob'    : prob[pred] if not is_binary else prob,
                'scores'  : score,
                'probs'   : prob,
                'labels'  : labels
            })
        return results
    
    @timer
    def predict(self, data, batch_size = 64, tqdm = lambda x: x, ** kwargs):
        if self.use_ea: self.ea_fn.fit(kwargs['samples'])
        if isinstance(data, pd.DataFrame): data = data.to_dict('records')
        elif not isinstance(data, list):   data = [data]

        results = []
        for i in tqdm(range(0, len(data), batch_size)):
            with time_logger.timer('pre-processing'):
                batch_infos = {
                    'ids'      : [d['id'] for d in data[i : i + batch_size]],
                    'channels' : [d['channels'] for d in data[i : i + batch_size]]
                }
                batch = self.get_input(data[i : i + batch_size], ** kwargs)
                if not self.use_fixed_length_input: batch = pad_batch(batch)
                batch = tf.cast(batch, tf.float32)

            with time_logger.timer('inference'):
                scores_per_channels = None
                if self.is_single_channel and len(batch_infos['channels'][0]) > 1:
                    labels, channel_scores  = self.infer_multi_channels(batch, ** batch_infos, ** kwargs)

                    complete_results = self._compute_metrics(labels, np.sum(channel_scores, axis = 0))
                    for b, res in enumerate(complete_results):
                        res['per_channel'] = {
                            ch : self._compute_metrics(labels, scores_per_channels[b : b + 1, i])[0]
                            for i, ch in enumerate(batch_infos['channels'][b])
                        }
                    results.extend(complete_results)
                else:
                    labels, scores  = self.infer(batch, ** batch_infos, ** kwargs)
                    results.extend(self._compute_metrics(labels, scores))
        
        return results

    def get_config_eeg(self, * args, ** kwargs):
        if self.use_ea:
            self.ea_fn.save_to_file(self.ea_file)
        if self.use_mel and not os.path.exists(self.mel_file):
            self.mel_fn.save_to_file(self.mel_file)
        if self.use_csp and not os.path.exists(self.csp_file):
            self.csp_fn.save_to_file(self.csp_file)

        return {
            'rate'    : self.rate,
            'channels'  : self.channels,

            'use_ea'    : self.use_ea,
            'mel_fn'    : self.mel_file,
            'csp_fn'    : self.csp_file,

            'max_input_length'       : self.max_input_length,
            'normalization_config'   : self.normalization_config,
            'use_fixed_length_input' : self.use_fixed_length_input,
            'keep_spatial_information' : self._keep_spatial_information
        }
