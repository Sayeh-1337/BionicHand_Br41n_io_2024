
# Copyright (C) 2022 yui-mhcp project's author, Langlois Quentin, UCLouvain, INGI. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import pad_batch, sample_df, knn
from custom_train_objects.losses import GE2ELoss
from custom_train_objects.generators import EEGGrouperGenerator
from models.interfaces.base_eeg_model import BaseEEGModel

logger = logging.getLogger(__name__)

class EEGEncoder(BaseEEGModel):
    input_signature = BaseEEGModel.eeg_signature
    get_input       = BaseEEGModel.get_eeg

    def __init__(self, rate, channels, *, distance_metric = 'euclidian', ** kwargs):
        self._init_eeg(rate, channels, ** kwargs)
        
        self.distance_metric = distance_metric
        
        super().__init__(** kwargs)
        
        self.model.train_step = self.train_step

    def init_train_config(self, ** kwargs):
        super().init_train_config(** kwargs)
        
        if hasattr(self.get_loss(), 'variables') and hasattr(self.get_metric(), 'set_variables'):
            self.get_metric().set_variables(self.get_loss().variables)

    def _build_model(self, embedding_dim = 32, final_activation = None, normalize = None, ** config):
        if normalize is None: normalize = self.distance_metric != 'cosine'
        if final_activation and normalize: final_activation = [final_activation, 'l2_normalize']
        config.update({
            'output_dim'       : embedding_dim,
            'final_activation' : final_activation
        })
        return super()._build_model(** config)

    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None, ), dtype = tf.string)

    def compile(self, loss = None, metrics = 'GE2EMetric', reduction = 'none', ** kwargs):
        if loss is None:
            loss = GE2ELoss(
                init_w          = 1. if self.distance_metric in ('dp', 'cosine') else -1.,
                reduction       = reduction,
                distance_metric = self.distance_metric,
                
            )
        if isinstance(metrics, str):
            metrics = [{'metric' : metrics, 'config' : {'name' : 'accuracy', 'distance_metric' : self.distance_metric}}]
        
        super(BaseEEGModel, self).compile(loss = loss, metrics = metrics, ** kwargs)

    def infer(self,
              inputs   = None,
              query    = None,
              samples  = None,
              training = False,

              ids      = None,
              channels = None,
              mask_by_id       = False,
              mask_by_channels = False,
              
              ** kwargs
             ):
        """
            Return the predicted label + score for each input

            Arguments :
                - inputs : `tf.Tensor` with shape `(B, t, n[, 1])`, the processed EEG signals
                - query  : `tf.Tensor` with shape `(B, embedding_dim)`, the embedded inputs EEG signals
        """
        assert inputs is not None or query is not None

        if not isinstance(samples, dict): samples = self.build_samples(samples)
        assert len(samples['embeddings']) > 0, "You must provide samples to classify new data !"
        
        if mask_by_channels and not self.is_single_channel:
            raise NotImplementedError('`mask_by_channels` is not supported for multi-channels models')
        
        if query is None: query = self(inputs, training = False)

        labels, scores = None, []
        if mask_by_channels:
            kwargs.update({'mask_by_id' : mask_by_id, 'mask_by_channels' : False})
            for q, id, ch in zip(query, ids, channels):
                mask        = samples['channels'] == ch
                sub_samples = {k : v[mask] for k, v in samples.items()}
                labels, s = self.infer(
                    query = q[tf.newaxis], ids = [id], channels = [ch], samples = sub_samples, ** kwargs
                )
                scores.append(s[0])
            return labels, np.array(scores)
        
        if mask_by_id:
            for q, id, ch in zip(query, ids, channels):
                mask        = samples['ids'] == id
                sub_samples = {k : v[mask] for k, v in samples.items()}
                labels, s = self.infer(
                    query = q[tf.newaxis], ids = [id], channels = [ch], samples = sub_samples, ** kwargs
                )
                scores.append(s[0])
            return labels, np.array(scores)
        
        return knn(
            query,
            samples['embeddings'],
            ids = samples['labels'],
            distance_metric = self.distance_metric,
            return_scores   = True,
            ** kwargs
        )

    def get_output(self, data):
        if isinstance(data, pd.DataFrame):        data = data['label' if 'label' in data.columns else 'id'].values
        elif isinstance(data, (dict, pd.Series)): data = data['label' if 'label' in data else 'id']
        return tf.as_string(data)

    def train_step(self, batch):
        inp, y_true = batch
        with tf.GradientTape() as tape:
            y_pred = self(inp, training = True)
            loss   = self.model.compute_loss(
                x = inp, y = y_true, y_pred = y_pred
            )

        #self.model._loss_tracker.update_state(loss)
        
        variables = self.model.trainable_weights + self.model.loss.variables
        grads = tape.gradient(loss, variables)
        self.model.optimizer.apply_gradients(zip(grads, variables))
        
        return self.model.compute_metrics(inp, y_true, y_pred, None)
        
    def fit(self, x, *, validation_data, generator_config = {}, ** kwargs):
        valid_generator_config = generator_config.copy()
        min_valid_utt = np.min(validation_data['label'].value_counts())
        if generator_config['n_utterance'] > min_valid_utt:
            while valid_generator_config['n_utterance'] > min_valid_utt:
                valid_generator_config['n_utterance'] //= 2
        
        train_generator = EEGGrouperGenerator(x, ** generator_config, shuffle = True)
        logger.info(str(train_generator))
        valid_generator = EEGGrouperGenerator(validation_data, ** valid_generator_config, shuffle = False)
        logger.info(str(valid_generator))
        
        kwargs.update({'shuffle_size' : 0, 'batch_size' : train_generator.batch_size, 'valid_batch_size' : valid_generator.batch_size})

        return super().fit(train_generator, validation_data = valid_generator, ** kwargs)
    
    def train(self, x, *, validation_data, generator_config = {}, ** kwargs):
        valid_generator_config = generator_config.copy()
        min_valid_utt = np.min(validation_data['label'].value_counts())
        if generator_config['n_utterance'] > min_valid_utt:
            while valid_generator_config['n_utterance'] > min_valid_utt:
                valid_generator_config['n_utterance'] //= 2
        
        train_generator = EEGGrouperGenerator(x, ** generator_config, shuffle = True)
        logger.info(str(train_generator))
        valid_generator = EEGGrouperGenerator(validation_data, ** valid_generator_config, shuffle = False)
        logger.info(str(valid_generator))
        
        kwargs.update({'shuffle_size' : 0, 'batch_size' : train_generator.batch_size, 'valid_batch_size' : valid_generator.batch_size})

        return super().train(train_generator, validation_data = valid_generator, ** kwargs)
    
    def predict(self, * args, samples = None, n_sample = None, ** kwargs):
        assert isinstance(samples, (dict, pd.DataFrame)), 'you must provide `samples` to compute the reference points'
        if n_sample is not None: samples = sample_df(samples, n = None, n_sample = n_sample)
        return super().predict(* args, samples = samples, ** kwargs)
        
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_eeg(),
            'distance_metric' : self.distance_metric
        })
        
        return config
