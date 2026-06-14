# Contains routines for labels creation, features extraction and normalization
#

import json

try:
    from cls_vid_features import VideoFeatures
except ImportError:
    VideoFeatures = None
from PIL import Image
import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
import joblib
from tqdm import tqdm
from multiprocessing import Pool
import time
import matplotlib.pyplot as plot
import librosa
plot.switch_backend('agg')
import shutil
import math
import wave
import contextlib
import cv2
import warnings

MISSING_DISTANCE_LABEL = -1.0


def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


def _distance_label_from_csv(distance_value_cm):
    distance_value_cm = float(distance_value_cm)
    if distance_value_cm < 0:
        return MISSING_DISTANCE_LABEL
    return distance_value_cm / 100.0


def _extract_file_feature_wrapper(args):
    obj, arg_in = args
    return obj.extract_file_feature(arg_in)


def _preprocess_file_feature_wrapper(args):
    obj, arg_in, spec_scaler = args
    return obj.preprocess_file_feature(arg_in, spec_scaler)


def _extract_file_label_wrapper(args):
    obj, arg_in = args
    return obj.extract_file_label(arg_in)


def _frame_stats_wrapper(file_path_and_hops):
    file_path, hop_len, label_hop_len = file_path_and_hops
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as wav_file:
            audio_len = wav_file.getnframes()
    except wave.Error:
        # Some generated FOA files are stored as IEEE float WAVs, which the stdlib wave
        # module cannot parse. scipy.io.wavfile can still read their shape reliably.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', getattr(wav, 'WavFileWarning', Warning))
            _, audio = wav.read(file_path, mmap=True)
        audio_len = int(audio.shape[0])
    nb_feat_frames = int(audio_len / float(hop_len))
    nb_label_frames = int(audio_len / float(label_hop_len))
    return file_name, nb_feat_frames, nb_label_frames


def _feature_stats_wrapper(feat_path):
    feat_file = np.load(feat_path, allow_pickle=True)
    if feat_file.dtype == object:
        feat_file = feat_file.item() if feat_file.ndim == 0 else np.vstack(feat_file)
    feat_file = np.asarray(feat_file, dtype=np.float64)
    frame_count = feat_file.shape[0]
    feat_sum = np.sum(feat_file, axis=0)
    feat_sq_sum = np.sum(np.square(feat_file), axis=0)
    return frame_count, feat_sum, feat_sq_sum


def _feature_shape_wrapper(args):
    feat_path, feature_label_resolution = args
    feat_file = np.load(feat_path, mmap_mode='r')
    file_name = os.path.basename(feat_path)
    frame_count = int(feat_file.shape[0])
    feature_dim = int(feat_file.shape[1])
    label_frames = frame_count // int(feature_label_resolution)
    return file_name, frame_count, label_frames, feature_dim


class FeatureClass:
    def __init__(self, params, is_eval=False):
        """

        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        self._vid_dir = os.path.join(self._dataset_dir, 'video_{}'.format('eval' if is_eval else 'dev'))
        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None
        self._vid_feat_dir = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(params.get('hop_len', int(self._fs * self._hop_len_s)))

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(params.get('label_hop_len', int(self._fs * self._label_hop_len_s)))
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = int(params.get('win_len', 2 * self._hop_len))
        self._nfft = int(params.get('n_fft', self._next_greater_power_of_2(self._win_len)))
        self._mel_fmin = float(params.get('mel_fmin', 0.0))
        self._mel_fmax = params.get('mel_fmax', None)

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 4

        self._multi_accdoa = params['multi_accdoa']
        self._use_salsalite = params['use_salsalite']
        self._foa_channel_order = params.get('foa_channel_order', 'WYZX').upper()
        self._split_manifest_path = params.get('split_manifest_path')
        self._norm_fit_splits = params.get('norm_fit_splits')
        self._num_workers = max(1, int(params.get('num_workers', 1)))
        self._feature_label_resolution = max(1, self._label_hop_len // self._hop_len)
        if self._use_salsalite and self._dataset in ('mic', 'foa'):
            # Initialize the spatial feature constants
            self._lower_bin = int(np.floor(params['fmin_doa_salsalite'] * self._nfft / float(self._fs)))
            self._lower_bin = np.max((1, self._lower_bin))
            self._upper_bin = int(np.floor(np.min((params['fmax_doa_salsalite'], self._fs//2)) * self._nfft / float(self._fs)))


            # Normalization factor for salsalite
            c = 343
            self._delta = 2 * np.pi * self._fs / (self._nfft * c)
            self._freq_vector = np.arange(self._nfft//2 + 1)
            self._freq_vector[0] = 1
            self._freq_vector = self._freq_vector[None, :, None]  # 1 x n_bins x 1

            # Initialize spectral feature constants
            if params.get('salsalite_nb_bins'):
                self._cutoff_bin = self._lower_bin + int(params['salsalite_nb_bins'])
            else:
                self._cutoff_bin = int(np.floor(params['fmax_spectra_salsalite'] * self._nfft / float(self._fs)))
            assert self._upper_bin <= self._cutoff_bin, 'Upper bin for doa feature {} is higher than cutoff bin for spectrogram {}!'.format(self._upper_bin, self._cutoff_bin)
            self._nb_mel_bins = self._cutoff_bin - self._lower_bin
        else:
            self._nb_mel_bins = params['nb_mel_bins']
            self._mel_wts = librosa.filters.mel(
                sr=self._fs,
                n_fft=self._nfft,
                n_mels=self._nb_mel_bins,
                fmin=self._mel_fmin,
                fmax=self._mel_fmax,
            ).T
        # Sound event classes dictionary
        self._nb_unique_classes = params['unique_classes']

        self._filewise_frames = {}

    def get_frame_stats(self):

        if len(self._filewise_frames) != 0:
            return

        audio_files = self._collect_files(self._aud_dir, '.wav')
        tqdm.write('Frame stats: {} files, {} workers'.format(len(audio_files), self._num_workers))
        arg_list = [(file_path, self._hop_len, self._label_hop_len) for file_path in audio_files]
        with Pool(processes=self._num_workers) as pool:
            for file_name, nb_feat_frames, nb_label_frames in tqdm(
                pool.imap(_frame_stats_wrapper, arg_list),
                total=len(arg_list),
                desc='Frame stats'
            ):
                self._filewise_frames[file_name] = [nb_feat_frames, nb_label_frames]
        return

    def _register_feature_frames(self, file_stem, nb_feat_frames, nb_label_frames=None):
        if nb_label_frames is None:
            nb_label_frames = nb_feat_frames // self._feature_label_resolution
        self._filewise_frames[file_stem] = [int(nb_feat_frames), int(nb_label_frames)]

    def _load_shape(self, file_path):
        return tuple(int(dim) for dim in np.load(file_path, mmap_mode='r').shape)

    def _load_audio(self, audio_path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', getattr(wav, 'WavFileWarning', Warning))
            fs, audio = wav.read(audio_path)
        if audio.ndim == 1:
            audio = audio[:, None]
        audio = audio[:, :self._nb_channels]
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            scale = float(max(abs(info.min), abs(info.max)))
            audio = audio.astype(np.float32) / scale
        else:
            audio = audio.astype(np.float32)
        audio = audio + self._eps
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input, _nb_frames):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T

    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self._eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1)) / 3.0)

        I_norm = I / E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0, 2, 1)), self._mel_wts), (0, 2, 1))
        foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv

    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def _get_salsalite(self, linear_spectra):
        # Adapted from the official SALSA repo- https://github.com/thomeou/SALSA
        if self._dataset == 'foa' and self._foa_channel_order == 'WYZX':
            # STARSS/DCASE FOA wavs are WYZX. SALSA uses channel 0 as W and
            # the remaining directional channels as X/Y/Z for phase features.
            linear_spectra = linear_spectra[:, :, [0, 3, 1, 2]]

        # spatial features
        phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        phase_vector = phase_vector / (self._delta * self._freq_vector)
        phase_vector = phase_vector[:, self._lower_bin:self._cutoff_bin, :]
        phase_vector[:, self._upper_bin:, :] = 0
        phase_vector = phase_vector.transpose((0, 2, 1)).reshape((phase_vector.shape[0], -1))

        # spectral features
        linear_spectra = np.abs(linear_spectra)**2
        for ch_cnt in range(linear_spectra.shape[-1]):
            linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10, top_db=None)
        linear_spectra = linear_spectra[:, self._lower_bin:self._cutoff_bin, :]
        linear_spectra = linear_spectra.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

        return np.concatenate((linear_spectra, phase_vector), axis=-1)

    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(audio_filename)

        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        # self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]

        audio_spec = self._spectrogram(audio_in, nb_feat_frames)
        return audio_spec, nb_feat_frames, nb_label_frames

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
        se_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        x_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                for active_event in active_event_list:
                    #print(active_event)
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]
                    dist_label[frame_ind, active_event[0]] = _distance_label_from_csv(active_event[5])

        label_mat = np.concatenate((se_label, x_label, y_label, z_label, dist_label), axis=1)
        return label_mat

    # OUTPUT LABELS
    def get_adpit_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
        """

        se_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))  # [nb_frames, 6, max_classes]
        x_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                active_event_list_per_class = []
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)
                    if i == len(active_event_list) - 1:  # if the last
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                            dist_label[frame_ind, 0, active_event_a0[0]] = _distance_label_from_csv(active_event_a0[5])
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            dist_label[frame_ind, 1, active_event_b0[0]] = _distance_label_from_csv(active_event_b0[5])
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                            dist_label[frame_ind, 2, active_event_b1[0]] = _distance_label_from_csv(active_event_b1[5])
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            dist_label[frame_ind, 3, active_event_c0[0]] = _distance_label_from_csv(active_event_c0[5])
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            dist_label[frame_ind, 4, active_event_c1[0]] = _distance_label_from_csv(active_event_c1[5])
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                            dist_label[frame_ind, 5, active_event_c2[0]] = _distance_label_from_csv(active_event_c2[5])

                    elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                            dist_label[frame_ind, 0, active_event_a0[0]] = _distance_label_from_csv(active_event_a0[5])
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            dist_label[frame_ind, 1, active_event_b0[0]] = _distance_label_from_csv(active_event_b0[5])
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                            dist_label[frame_ind, 2, active_event_b1[0]] = _distance_label_from_csv(active_event_b1[5])
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            dist_label[frame_ind, 3, active_event_c0[0]] = _distance_label_from_csv(active_event_c0[5])
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            dist_label[frame_ind, 4, active_event_c1[0]] = _distance_label_from_csv(active_event_c1[5])
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                            dist_label[frame_ind, 5, active_event_c2[0]] = _distance_label_from_csv(active_event_c2[5])
                        active_event_list_per_class = []

        label_mat = np.stack((se_label, x_label, y_label, z_label, dist_label), axis=2)  # [nb_frames, 6, 5(=act+XYZ+dist), max_classes]
        return label_mat

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------

    def extract_file_feature(self, _arg_in):
        _file_cnt, _wav_path, _feat_path = _arg_in
        if os.path.exists(_feat_path):
            try:
                feat = np.load(_feat_path)
                if self._use_salsalite:
                    expected_channels = (2 * self._nb_channels) - 1
                else:
                    expected_channels = 7 if self._dataset == 'foa' else (self._nb_channels + nCr(self._nb_channels, 2))
                expected_dim = self._nb_mel_bins * expected_channels
                if feat.ndim != 2 or feat.shape[1] != expected_dim:
                    raise ValueError('Invalid cached feature shape {} for {}'.format(feat.shape, _feat_path))
                feat = np.asarray(feat, dtype=np.float64)
                file_stem = os.path.splitext(os.path.basename(_feat_path))[0]
                nb_feat_frames = feat.shape[0]
                nb_label_frames = nb_feat_frames // self._feature_label_resolution
                return file_stem, nb_feat_frames, nb_label_frames, feat.shape[0], np.sum(feat, axis=0), np.sum(np.square(feat), axis=0)
            except Exception:
                # A previously interrupted run can leave behind truncated or malformed .npy files.
                # Remove them and rebuild the feature from the source wav.
                if os.path.exists(_feat_path):
                    os.unlink(_feat_path)
        spect, nb_feat_frames, nb_label_frames = self._get_spectrogram_for_file(_wav_path)

        # extract mel
        if not self._use_salsalite:
            mel_spect = self._get_mel_spectrogram(spect)

        feat = None
        if self._dataset == 'foa':
            if self._use_salsalite:
                feat = self._get_salsalite(spect)
            else:
                # extract intensity vectors
                foa_iv = self._get_foa_intensity_vectors(spect)
                feat = np.concatenate((mel_spect, foa_iv), axis=-1)
        elif self._dataset == 'mic':
            if self._use_salsalite:
                feat = self._get_salsalite(spect)
            else:
                # extract gcc
                gcc = self._get_gcc(spect)
                feat = np.concatenate((mel_spect, gcc), axis=-1)
        else:
            raise ValueError('Unknown dataset format {}'.format(self._dataset))

        if feat is not None:
            np.save(_feat_path, feat)
            feat = np.asarray(feat, dtype=np.float64)
            file_stem = os.path.splitext(os.path.basename(_feat_path))[0]
            return file_stem, nb_feat_frames, nb_label_frames, feat.shape[0], np.sum(feat, axis=0), np.sum(np.square(feat), axis=0)

    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)
        audio_files = self._collect_files(self._aud_dir, '.wav')
        tqdm.write('Extracting features -> {} ({} files, {} workers)'.format(self._feat_dir, len(audio_files), self._num_workers))
        arg_list = [
            (file_cnt, wav_path, os.path.join(self._feat_dir, '{}.npy'.format(os.path.splitext(os.path.basename(wav_path))[0])))
            for file_cnt, wav_path in enumerate(audio_files)
        ]

        total_frames = 0
        feat_sum = None
        feat_sq_sum = None
        with Pool(processes=self._num_workers) as pool:
            for result in tqdm(
                pool.imap(_extract_file_feature_wrapper, [(self, arg) for arg in arg_list]),
                total=len(arg_list),
                desc='Extracting features'
            ):
                if result is None:
                    continue
                file_stem, nb_feat_frames, nb_label_frames, frame_count, cur_sum, cur_sq_sum = result
                self._register_feature_frames(file_stem, nb_feat_frames, nb_label_frames)
                total_frames += frame_count
                if feat_sum is None:
                    feat_sum = cur_sum
                    feat_sq_sum = cur_sq_sum
                else:
                    feat_sum += cur_sum
                    feat_sq_sum += cur_sq_sum
        return {
            'total_frames': total_frames,
            'feat_sum': feat_sum,
            'feat_sq_sum': feat_sq_sum,
        }

    def preprocess_file_feature(self, _arg_in, spec_scaler):
        file_name, feat_dir, feat_dir_norm = _arg_in
        feat_file = np.load(os.path.join(feat_dir, file_name))
        feat_file = spec_scaler.transform(feat_file)
        np.save(
            os.path.join(feat_dir_norm, file_name),
            feat_file
        )

    def _get_norm_fit_feature_files(self, all_feat_files):
        if not self._norm_fit_splits:
            return all_feat_files
        if not self._split_manifest_path or not os.path.exists(self._split_manifest_path):
            raise FileNotFoundError('norm_fit_splits requires split_manifest_path, got {}'.format(self._split_manifest_path))
        with open(self._split_manifest_path, 'r') as handle:
            manifest = json.load(handle)
        split_names = self._norm_fit_splits
        if isinstance(split_names, str):
            split_names = [split_names]
        selected_stems = set()
        for split_name in split_names:
            split_entry = manifest.get('splits', {}).get(split_name)
            if split_entry is None:
                raise KeyError('Split {} not found in {}'.format(split_name, self._split_manifest_path))
            selected_stems.update(split_entry.get('stems', []))
        selected_files = [
            file_name for file_name in all_feat_files
            if os.path.splitext(file_name)[0] in selected_stems
        ]
        if not selected_files:
            raise ValueError('No feature files matched norm_fit_splits={} in {}'.format(split_names, self._feat_dir))
        tqdm.write('Fitting normalization on splits {}: {} / {} files'.format(
            ','.join(split_names), len(selected_files), len(all_feat_files)
        ))
        return selected_files

    def preprocess_features(self, stats_cache=None):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            tqdm.write('Loaded normalization weights from {}'.format(normalized_features_wts_file))

        else:
            if stats_cache is not None and stats_cache.get('total_frames') and not self._norm_fit_splits:
                total_frames = stats_cache['total_frames']
                feat_sum = stats_cache['feat_sum']
                feat_sq_sum = stats_cache['feat_sq_sum']
                tqdm.write('Using cached feature statistics from extraction pass')
            else:
                all_feat_files = sorted(os.listdir(self._feat_dir))
                fit_feat_files = self._get_norm_fit_feature_files(all_feat_files)
                feat_paths = [os.path.join(self._feat_dir, file_name) for file_name in fit_feat_files]
                tqdm.write('Estimating normalization weights from {} ({} files, {} workers)'.format(
                    self._feat_dir, len(feat_paths), self._num_workers))
                total_frames = 0
                feat_sum = None
                feat_sq_sum = None
                with Pool(processes=self._num_workers) as pool:
                    for frame_count, cur_sum, cur_sq_sum in tqdm(
                        pool.imap(_feature_stats_wrapper, feat_paths),
                        total=len(feat_paths),
                        desc='Estimating weights'
                    ):
                        total_frames += frame_count
                        if feat_sum is None:
                            feat_sum = cur_sum
                            feat_sq_sum = cur_sq_sum
                        else:
                            feat_sum += cur_sum
                            feat_sq_sum += cur_sq_sum

            if total_frames == 0:
                raise ValueError('No feature frames found in {}'.format(self._feat_dir))

            mean = feat_sum / total_frames
            var = np.maximum((feat_sq_sum / total_frames) - np.square(mean), self._eps)
            spec_scaler = preprocessing.StandardScaler()
            spec_scaler.mean_ = mean
            spec_scaler.var_ = var
            spec_scaler.scale_ = np.sqrt(var)
            spec_scaler.n_features_in_ = mean.shape[0]
            spec_scaler.n_samples_seen_ = total_frames
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            tqdm.write('Saved normalization weights to {}'.format(normalized_features_wts_file))

        all_feat_files = sorted(os.listdir(self._feat_dir))
        arg_list = [(file_name, self._feat_dir, self._feat_dir_norm) for file_name in all_feat_files]
        tqdm.write('Normalizing features -> {} ({} files, {} workers)'.format(
            self._feat_dir_norm, len(arg_list), self._num_workers))

        with Pool(processes=self._num_workers) as pool:
            list(tqdm(
                pool.imap(_preprocess_file_feature_wrapper, [(self, arg, spec_scaler) for arg in arg_list]),
                total=len(arg_list),
                desc='Normalizing features'
            ))

        tqdm.write('Normalized features written to {}'.format(self._feat_dir_norm))

    def write_dataset_stats_cache(self, force_rebuild=False):
        stats_path = self.get_dataset_stats_file()
        feat_dir = self.get_normalized_feat_dir()
        label_dir = self.get_label_dir()

        if not os.path.isdir(feat_dir):
            raise ValueError('Normalized feature directory not found: {}'.format(feat_dir))

        feat_files = sorted([name for name in os.listdir(feat_dir) if name.endswith('.npy')])
        if not feat_files:
            raise ValueError('No normalized features found in {}'.format(feat_dir))

        if force_rebuild or not self._filewise_frames:
            tqdm.write('Building dataset stats cache -> {} ({} files, {} workers)'.format(
                stats_path, len(feat_files), self._num_workers))
            arg_list = [
                (os.path.join(feat_dir, file_name), self._feature_label_resolution)
                for file_name in feat_files
            ]
            feature_dim = None
            with Pool(processes=self._num_workers) as pool:
                for file_name, feat_frames, label_frames, current_feature_dim in tqdm(
                    pool.imap(_feature_shape_wrapper, arg_list),
                    total=len(arg_list),
                    desc='Caching dataset stats'
                ):
                    self._register_feature_frames(os.path.splitext(file_name)[0], feat_frames, label_frames)
                    if feature_dim is None:
                        feature_dim = current_feature_dim
        else:
            feature_dim = self._load_shape(os.path.join(feat_dir, feat_files[0]))[1]

        label_shape = None
        if not self._is_eval and label_dir and os.path.isdir(label_dir):
            label_files = sorted([name for name in os.listdir(label_dir) if name.endswith('.npy')])
            if label_files:
                label_shape = self._load_shape(os.path.join(label_dir, label_files[0]))

        stats_payload = {
            'version': 1,
            'dataset_combination': self._dataset_combination,
            'feature_label_resolution': int(self._feature_label_resolution),
            'nb_mel_bins': int(self._nb_mel_bins),
            'nb_ch': int(feature_dim // self._nb_mel_bins),
            'feature_dim': int(feature_dim),
            'multi_accdoa': bool(self._multi_accdoa),
            'label_shape': list(label_shape) if label_shape is not None else None,
            'files': {
                '{}.npy'.format(file_stem): {
                    'feat_frames': int(file_frames[0]),
                    'label_frames': int(file_frames[1]),
                }
                for file_stem, file_frames in sorted(self._filewise_frames.items())
            }
        }

        with open(stats_path, 'w') as handle:
            json.dump(stats_payload, handle, indent=2, sort_keys=True)
        tqdm.write('Saved dataset stats cache to {}'.format(stats_path))

    def extract_file_label(self, _arg_in):
        file_name, loc_desc_folder, label_dir = _arg_in
        wav_filename = '{}.wav'.format(file_name.split('.')[0])
        nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]
        desc_file_polar = self.load_output_format_file(os.path.join(loc_desc_folder, file_name))
        desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
        if self._multi_accdoa:
            label_mat = self.get_adpit_labels_for_file(desc_file, nb_label_frames)
        else:
            label_mat = self.get_labels_for_file(desc_file, nb_label_frames)
        np.save(os.path.join(label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self.get_frame_stats()
        self._label_dir = self.get_label_dir()
        create_folder(self._label_dir)

        metadata_files = self._collect_files(self._desc_dir, '.csv')
        tqdm.write('Extracting labels -> {} ({} files, {} workers)'.format(
            self._label_dir, len(metadata_files), self._num_workers))
        arg_list = [
            (os.path.basename(file_path), os.path.dirname(file_path), self._label_dir)
            for file_path in metadata_files
        ]

        with Pool(processes=self._num_workers) as pool:
            list(tqdm(
                pool.imap(_extract_file_label_wrapper, [(self, arg) for arg in arg_list]),
                total=len(arg_list),
                desc='Extracting labels'
            ))

    # ------------------------------- EXTRACT VISUAL FEATURES AND PREPROCESS IT -------------------------------
    @staticmethod
    def _read_vid_frames(vid_filename):
        cap = cv2.VideoCapture(vid_filename)
        pil_frames = []
        frame_cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_cnt % 3 == 0:
                resized_frame = cv2.resize(frame, (360, 180))
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                pil_frames.append(pil_frame)
            frame_cnt += 1
        cap.release()
        cv2.destroyAllWindows()

        return pil_frames

    def extract_file_vid_feature(self, _arg_in):
        _file_cnt, _mp4_path, _vid_feat_path = _arg_in
        vid_feat = None

        vid_frames = self._read_vid_frames(_mp4_path)
        pretrained_vid_model = VideoFeatures()
        vid_feat = pretrained_vid_model(vid_frames)
        vid_feat = np.array(vid_feat)

        if vid_feat is not None:
            np.save(_vid_feat_path, vid_feat)

    def extract_visual_features(self):
        self._vid_feat_dir = self.get_vid_feat_dir()
        create_folder(self._vid_feat_dir)
        video_files = self._collect_files(self._vid_dir, '.mp4')
        tqdm.write('Extracting visual features -> {} ({} files)'.format(self._vid_feat_dir, len(video_files)))
        for file_cnt, mp4_path in enumerate(tqdm(video_files, desc='Extracting visual features')):
            vid_feat_path = os.path.join(self._vid_feat_dir, '{}.npy'.format(os.path.splitext(os.path.basename(mp4_path))[0]))
            self.extract_file_vid_feature((file_cnt, mp4_path, vid_feat_path))

    @staticmethod
    def _collect_files(root_dir, suffix):
        collected_files = []
        for current_root, _, file_names in os.walk(root_dir):
            for file_name in file_names:
                if file_name.endswith(suffix):
                    collected_files.append(os.path.join(current_root, file_name))
        return sorted(collected_files)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file, cm2m=False):  # TODO: Reconsider cm2m conversion
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        _words = []     # For empty files
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 4:  # frame, class idx,  polar coordinates(2) # no distance data, for example in eval pred
                _output_dict[_frame_ind].append([int(_words[1]), 0, float(_words[2]), float(_words[3])])
            if len(_words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and 2
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            if len(_words) == 6: # frame, class idx, source_id, polar coordinates(2), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])/100 if cm2m else float(_words[5])])
            elif len(_words) == 7: # frame, class idx, source_id, cartesian coordinates(3), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5]), float(_words[6])/100 if cm2m else float(_words[6])])
        _fid.close()
        if len(_words) == 7:
            _output_dict = self.convert_output_format_cartesian_to_polar(_output_dict)
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), float(_value[4])))
                # TODO: What if our system estimates track cound and distence (or only one of them)
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames / float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt + self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

    def organize_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in every frame, similar to segment_labels but at frame level
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each frame
                dictionary_name[frame-index][class-index][track-index] = [azimuth, elevation, (distance)] or
                                                                         [x, y, z, (distance)]
        '''
        nb_frames = _max_frames
        output_dict = {x: {} for x in range(nb_frames)}
        for frame_idx in range(0, _max_frames):
            if frame_idx not in _pred_dict:
                continue
            for [class_idx, track_idx, *localization] in _pred_dict[frame_idx]:
                if class_idx not in output_dict[frame_idx]:
                    output_dict[frame_idx][class_idx] = {}

                if track_idx not in output_dict[frame_idx][class_idx]:
                    output_dict[frame_idx][class_idx][track_idx] = localization
                else:
                    # Repeated track_idx for the same class_idx in the same frame_idx, the model is not estimating
                    # track IDs, so track_idx is set to a negative value to distinguish it from a proper track ID
                    min_track_idx = np.min(np.array(list(output_dict[frame_idx][class_idx].keys())))
                    new_track_idx = min_track_idx - 1 if min_track_idx < 0 else -1
                    output_dict[frame_idx][class_idx][new_track_idx] = localization

        return output_dict

    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = self._nb_unique_classes
        _is_polar = _doa_labels.shape[-1] == 2*_nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2*_nb_classes]
            _z = _doa_labels[:, 2*_nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class], _z[_frame_ind, _tmp_class]])
        return _output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180.

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append(tmp_val[0:2] + [x, y, z] + tmp_val[4:])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                    r = np.sqrt(x**2 + y**2 + z**2)
                    out_dict[frame_cnt].append(tmp_val[0:2] + [azimuth, elevation] + tmp_val[5:])
        return out_dict

    # ------------------------------- Misc public functions -------------------------------

    def get_normalized_feat_dir(self):
        feature_suffix = '{}_salsa'.format(self._dataset_combination) if self._use_salsalite else self._dataset_combination
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format(feature_suffix)
        )

    def get_unnormalized_feat_dir(self):
        feature_suffix = '{}_salsa'.format(self._dataset_combination) if self._use_salsalite else self._dataset_combination
        return os.path.join(
            self._feat_label_dir,
            '{}'.format(feature_suffix)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir,
               '{}_label'.format('{}_adpit'.format(self._dataset_combination) if self._multi_accdoa else self._dataset_combination)
        )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_dataset_stats_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_dataset_stats.json'.format(self._dataset_combination)
        )

    def get_vid_feat_dir(self):
        return os.path.join(self._feat_label_dir, 'video_{}'.format('eval' if self._is_eval else 'dev'))

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_mel_bins(self):
        return self._nb_mel_bins


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        tqdm.write('Creating folder {}'.format(folder_name))
        os.makedirs(folder_name)


def delete_and_create_folder(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)
