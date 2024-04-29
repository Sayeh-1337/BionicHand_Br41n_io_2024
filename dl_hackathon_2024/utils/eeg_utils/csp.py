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

'''	Functions used for common spatial patterns'''

import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import linalg
from scipy.special import binom

from loggers import timer
from utils import sample_df, convert_to_str
from utils.file_utils import load_data, load_json, dump_data, dump_json
from utils.eeg_utils.filters import load_filterbank, butter_fir_filter

logger = logging.getLogger(__name__)

DEFAULT_FILTERBANK  = {
    'bandwidth' : np.array([2, 4, 8, 16, 32]),
    'order'     : 2,
    'max_freq'  : 40,
    'ftype'     : 'butter'
}
DEFAULT_WINDOWS = np.array([
    [2.5,3.5],
    [3,4],
    [3.5,4.5],
    [4,5],
    [4.5,5.5],
    [5,6],
    [2.5,4.5],
    [3,5],
    [3.5,5.5],
    [4,6],
    [2.5,6]
], dtype = np.float32)


class CSP(object):
    def __init__(self,
                 sampling_rate,
                 n_csp = 4,
                 filter_bank    = DEFAULT_FILTERBANK,
                 time_windows   = DEFAULT_WINDOWS,
                 normalize_mode = None,
                 pre_emph       = 0.,
                 _filter_bank   = None,
                 csp_mapping    = None
                ):
        if not isinstance(filter_bank, dict) and len(filter_bank.shape) == 1:
            filter_bank = {'bandwidth' : filter_bank}
        
        if isinstance(time_windows, list): time_windows = np.array(time_windows)
        if isinstance(_filter_bank, list): _filter_bank = np.array(_filter_bank)

        self.n_csp = n_csp
        self.sampling_rate  = sampling_rate
        self.filter_bank_config    = filter_bank if isinstance(filter_bank, dict) else None
        self.time_windows   = time_windows
        self.normalize_mode = normalize_mode
        self.pre_emph       = pre_emph
        
        if self.time_windows.dtype == np.float32:
            self.time_windows = (self.time_windows * self.sampling_rate).astype(np.int32)
        
        self.filter_bank    = load_filterbank(
            fs = sampling_rate, ** filter_bank
        ) if _filter_bank is None else _filter_bank
        self.csp_mapping    = {} if csp_mapping is None else csp_mapping
    
    @property
    def rate(self):
        return self.sampling_rate
    
    @property
    def n_bands(self):
        return self.filter_bank.shape[0]
    
    @property
    def n_windows(self):
        return self.time_windows.shape[0]
    
    @property
    def n_features(self):
        return self.n_csp * self.n_bands * self.n_windows

    def __str__(self):
        config = self.get_config()
        des = "\n========== {} ==========\n".format(config.pop('class_name'))
        des += '- {}\t: {}\n'.format('# subjects', len(self.csp_mapping))
        des += '- {}\t: {}\n'.format('# bands', self.n_bands)
        des += '- {}\t: {}\n'.format('# windows', self.n_windows)
        des += '- {}\t: {}\n'.format('# features', self.n_features)
        for k, v in config.items():
            if k.startswith('_'): continue
            des += "- {}\t: {}\n".format(k, v)
        return des

    def __call__(self, trials, subj_id):
        if len(tf.shape(trials)) == 2: trials = tf.expand_dims(trials, axis = 0)
        trials = tf.cast(trials, tf.float32)
        
        if self.pre_emph > 0.:
            trials = tf.concat([
                trials[:, :, :1],
                trials[:, :, 1:] - self.pre_emph * trials[:, :, :-1]
            ], axis = -1)
        
        return self.compute_features(trials, subj_id = subj_id)
    
    def __contains__(self, subj_id):
        return subj_id in self.csp_mapping
    
    def __getitem__(self, subj_id):
        return self.csp_mapping[subj_id]
    
    def __setitem__(self, subj_id, w):
        self.csp_mapping[subj_id] = w
    
    def fit(self, x, y, subj_id, overwrite = False):
        subj_id = convert_to_str(subj_id)
        if overwrite or subj_id not in self:
            assert y is not None, 'You must provide label to the 1st call for a new subject : {} !'.format(subj_id)
            logger.info('Building projection for subject {} with {} trials and {} labels !'.format(
                subj_id, len(x), len(np.unique(y))
            ))
            self[subj_id] = generate_projection(x, y, self.n_csp, self.filter_bank, self.time_windows)
        return self[subj_id]
    
    def fit_dataset(self, dataset, id_column = 'id', n_sample = None, ** kwargs):
        assert isinstance(dataset, pd.DataFrame)
        
        for subj_id, data in dataset.groupby(id_column):
            if n_sample:
                data = sample_df(
                    data, on = 'label', n = None, n_sample = n_sample, random_state = 10
                )
            trials = np.array(list(data['eeg'].values))
            self.fit(trials, data['label'].values, subj_id, ** kwargs)
            
    
    def _compute_features(self, trials, subj_id, labels = None):
        w = self.fit(trials, labels, subj_id)
        return extract_feature(trials, w, self.filter_bank, self.time_windows).astype(np.float32)

    def compute_features(self, trials, subj_id, labels = None):
        if len(tf.shape(trials)) == 2: trials = tf.expand_dims(trials, axis = 0)
        features = tf.numpy_function(
            self._compute_features, [trials, subj_id], Tout = tf.float32
        )
        features.set_shape([None, self.n_features])
        return features
    
    def normalize(self, mel):
        if self.normalize_mode is None:
            return mel
        elif self.normalize_mode == 'per_feature':
            mean = tf.reduce_mean(mel, axis = 1, keepdims = True)
            std = tf.math.reduce_std(mel, axis = 1, keepdims = True)
        elif self.normalize_mode == 'all_feature':
            mean = tf.reduce_mean(mel)
            std = tf.math.reduce_std(mel)
            
        return (mel - mean) / (std + 1e-5)
    
    def get_config(self):
        return {
            'class_name'    : self.__class__.__name__,
            'n_csp'         : self.n_csp,
            'sampling_rate'     : self.sampling_rate,
            'filter_bank'       : self.filter_bank_config,
            'time_windows'      : self.time_windows,
            'normalize_mode'    : self.normalize_mode,
            'pre_emph'      : self.pre_emph,
            '_filter_bank'  : self.filter_bank
        }
    
    def save_to_file(self, filename):
        if '.json' not in filename: filename += '.json'
        
        dump_json(filename, self.get_config(), indent = 4)
        dump_data(filename.replace('.json', '.pkl'), self.csp_mapping)

        return filename
    
    @classmethod
    def load_from_file(cls, filename):
        config     = load_json(filename)
        class_name = config.pop('class_name')
        return cls(** config, csp_mapping = load_data(filename.replace('.json', '.pkl')))

@timer
def csp_one_one(cov_matrix, nb_csp):
    '''	
    calculate spatial filter for class all pairs of classes 

    Keyword arguments:
    cov_matrix -- numpy array of size [NO_channels, NO_channels]
    NO_csp -- number of spatial filters (24)

    Return:	spatial filter numpy array of size [22, nb_csp] 
    '''
    nb_classes, nb_channels = cov_matrix.shape[:2]
    
    n_comb = binom(nb_classes, 2)

    nb_filtpairs = int(nb_csp / (n_comb * 2))
    
    w = np.zeros((nb_channels, nb_csp))
    
    idx = 0 # internal counter 
    for c1 in range(nb_classes):
        for c2 in range(c1 + 1, nb_classes):
            w[:, nb_filtpairs * 2 * idx : nb_filtpairs * 2 * (idx + 1)] = gevd(cov_matrix[c1], cov_matrix[c2], nb_filtpairs)
            idx +=1
    return w 

@timer
def generate_projection(data, labels, nb_csp, filter_bank, time_windows): 
    '''	generate spatial filters for every timewindow and frequancy band

    Keyword arguments:
    data -- numpy array of size [NO_trials,channels,time_samples]
    class_vec -- containing the class labels, numpy array of size [NO_trials]
    NO_csp -- number of spatial filters (24)
    filter_bank -- numpy array containing butter sos filter coeffitions dim  [NO_bands,order,6]
    time_windows -- numpy array [[start_time1,end_time1],...,[start_timeN,end_timeN]] 

    Return:	spatial filter numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp] 
    '''
    import pyriemann.utils.mean as rie_mean

    time_windows = np.reshape(time_windows, [-1, 2])
    
    classes     = sorted(np.unique(labels))
    nb_classes  = len(classes)
    nb_bands    = filter_bank.shape[0]
    nb_windows  = time_windows.shape[0]
    nb_channels = data.shape[1]
    nb_trials   = data.shape[0]

    cov_avg    = np.zeros((nb_classes, nb_channels, nb_channels))
    # Initialize spatial filter: 
    w = np.zeros((nb_windows, nb_bands, nb_channels, nb_csp))
    for win_idx, (start, end) in enumerate(time_windows):
        for subband in range(nb_bands):
            filtered   = butter_fir_filter(data[:, :, start : end], filter_bank[subband])
            covariance = np.matmul(filtered, np.transpose(filtered, [0, 2, 1]))
            # calculate average of covariance matrix 
            for c in range(nb_classes):
                cov_avg[c, :, :] = rie_mean.mean_covariance(covariance[labels == classes[c]], metric = 'euclid')
            w[win_idx, subband, :, :] = csp_one_one(cov_avg, nb_csp) 
    return w

def generate_eye(data, labels, filter_bank, time_windows): 
    '''	generate unity spatial filters for every timewindow and frequancy band

    Keyword arguments:
    data -- numpy array of size [NO_trials,channels,time_samples]
    class_vec -- containing the class labels, numpy array of size [NO_trials]
    filter_bank -- numpy array containing butter sos filter coeffitions dim  [NO_bands,order,6]
    time_windows -- numpy array [[start_time1,end_time1],...,[start_timeN,end_timeN]] 

    Return:	spatial unity filter numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp] 
    '''
    time_windows = np.reshape(time_windows, [-1, 2])
    
    nb_bands    = filter_bank.shape[0]
    nb_windows  = time_windows.shape[0]
    nb_channels = data.shape[1]
    nb_trials   = data.shape[0]

    # Initialize spatial filter: 
    eye = np.eye(nb_channels)
    
    w = np.zeros((nb_windows, nb_bands, nb_channels, nb_channels))
    for i in range(nb_windows):
        for j in range(nb_bands):
            w[i, j] = eye
    return w

@timer
def extract_feature(data, w, filter_bank, time_windows):
    '''	calculate features using the precalculated spatial filters

    Keyword arguments:
    data -- numpy array of size [NO_trials,channels,time_samples]
    w -- spatial filters, numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp]
    filter_bank -- numpy array containing butter sos filter coeffitions dim  [NO_bands,order,6]
    time_windows -- numpy array [[start_time1,end_time1],...,[start_timeN,end_timeN]] 

    Return:	features, numpy array of size [NO_trials,(NO_csp*NO_bands*NO_time_windows)] 
    '''
    
    time_windows = np.reshape(time_windows, [-1, 2])
    
    nb_csp      = w.shape[-1]
    nb_bands    = filter_bank.shape[0]
    nb_windows  = time_windows.shape[0]
    nb_channels = data.shape[1]
    nb_trials   = data.shape[0]

    nb_features = nb_csp * nb_bands * nb_windows
	
    feature_mat = np.zeros((nb_trials, nb_windows, nb_bands, nb_csp))
    for win_idx, (start, end) in enumerate(time_windows):
        for subband in range(nb_bands):
            data_s   = np.matmul(np.transpose(w[win_idx, subband]), data[:, :, start : end])
            filtered = butter_fir_filter(data_s, filter_bank[subband])

            feature_mat[:, win_idx, subband] = np.var(filtered, axis = 2)
    return np.reshape(np.log10(feature_mat), (nb_trials, -1))


'''	general eigenvalue decomposition'''

@timer
def gevd(x1, x2, nb_pairs):
    '''Solve generalized eigenvalue decomposition
    
    Keyword arguments:
    x1 -- numpy array of size [NO_channels, NO_samples]
    x2 -- numpy array of size [NO_channels, NO_samples]
    no_pairs -- number of pairs of eigenvectors to be returned 

    Return:	numpy array of 2*No_pairs eigenvectors 
    '''
    ev, vr = linalg.eig(x1, x2, right = True) 

    sort_indices   = np.argsort(np.abs(ev))
    indexes        = np.zeros(2 * nb_pairs, dtype = np.int32)
    indexes[0 : nb_pairs]            = sort_indices[0 : nb_pairs]
    indexes[nb_pairs : 2 * nb_pairs] = sort_indices[-nb_pairs :]
    
    return vr[:, indexes]