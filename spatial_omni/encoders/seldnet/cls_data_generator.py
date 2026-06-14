#
# Data generator for training the SELDnet
#

import os
import json
import re
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random
from tqdm import tqdm


class DataGenerator(object):
    def __init__(self, params, split=1, shuffle=True, per_file=False, is_eval=False, selected_files=None):

        self._per_file = per_file
        self._is_eval = is_eval
        self._split_selector = split
        self._selected_files = set(selected_files) if selected_files is not None else None
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()
        self._multi_accdoa = params['multi_accdoa']
        self._split_manifest = self._load_split_manifest(params.get('split_manifest_path'))
        self._selected_stems = self._resolve_selected_stems()
        self._dataset_stats = self._load_dataset_stats()

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._nb_classes = self._feat_cls.get_nb_classes()

        self._circ_buf_feat = None
        self._circ_buf_label = None

        self._modality = params['modality']
        if self._modality == 'audio_visual':
            self._vid_feature_seq_len = self._label_seq_len  # video feat also at 10 fps same as label resolutions (100ms)
            self._vid_feat_dir = self._feat_cls.get_vid_feat_dir()
            self._circ_buf_vid_feat = None

        self._get_filenames_list_and_feat_label_sizes()

        tqdm.write(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        tqdm.write(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
        if self._is_eval:
            label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3*4)
        else:
            if self._multi_accdoa is True:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3*4)
            else:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*4)

        if self._modality == 'audio_visual':
            vid_feat_shape = (self._batch_size, self._vid_feature_seq_len, 7, 7)
            return feat_shape, vid_feat_shape, label_shape
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    @staticmethod
    def _normalize_split_selector(split):
        if isinstance(split, np.ndarray):
            split = split.tolist()
        if isinstance(split, (list, tuple, set)):
            return list(split)
        return [split]

    @staticmethod
    def _load_split_manifest(manifest_path):
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, 'r') as handle:
                return json.load(handle)
        return None

    def _resolve_selected_stems(self):
        if not self._split_manifest:
            return None
        selected_stems = set()
        for selector in self._normalize_split_selector(self._split_selector):
            if isinstance(selector, str):
                split_name = selector.lower()
                if split_name in self._split_manifest.get('splits', {}):
                    selected_stems.update(self._split_manifest['splits'][split_name]['stems'])
        return selected_stems if selected_stems else None

    def _load_dataset_stats(self):
        stats_path = self._feat_cls.get_dataset_stats_file()
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as handle:
                return json.load(handle)
        return None

    @staticmethod
    def _extract_fold_index(filename):
        match = re.match(r'^fold(\d+)(?:_|$)', filename)
        return int(match.group(1)) if match else None

    def _is_selected_filename(self, filename):
        if self._selected_files is not None:
            return filename in self._selected_files
        if self._is_eval and self._selected_stems is None:
            return True
        stem = os.path.splitext(filename)[0]
        if self._selected_stems is not None:
            return stem in self._selected_stems

        selectors = self._normalize_split_selector(self._split_selector)
        numeric_selectors = set()
        for selector in selectors:
            try:
                numeric_selectors.add(int(selector))
            except (TypeError, ValueError):
                continue
        fold_index = self._extract_fold_index(filename)
        return fold_index in numeric_selectors if fold_index is not None else False

    def _configure_label_shapes(self):
        temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]), mmap_mode='r')
        if self._multi_accdoa is True:
            self._num_track_dummy = temp_label.shape[-3]
            self._num_axis = temp_label.shape[-2]
            self._num_class = temp_label.shape[-1]
        else:
            self._label_len = temp_label.shape[-1]
        self._doa_len = 3

    def _get_filenames_list_and_feat_label_sizes_from_cache(self):
        if not self._dataset_stats or 'files' not in self._dataset_stats:
            return False

        tqdm.write('Loading dataset stats cache')
        files_meta = self._dataset_stats['files']
        max_frames, total_frames = -1, 0
        self._filenames_list = []

        for filename in sorted(files_meta):
            if not os.path.exists(os.path.join(self._feat_dir, filename)):
                continue
            if not self._is_selected_filename(filename):
                continue
            if self._modality != 'audio' and (not hasattr(self, '_vid_feat_dir') or not os.path.exists(os.path.join(self._vid_feat_dir, filename))):
                continue

            feat_frames = int(files_meta[filename]['feat_frames'])
            self._filenames_list.append(filename)
            total_frames += (feat_frames - (feat_frames % self._feature_seq_len))
            if feat_frames > max_frames:
                max_frames = feat_frames

        if not self._filenames_list:
            return False

        self._nb_frames_file = max_frames if self._per_file else int(files_meta[self._filenames_list[0]]['feat_frames'])
        self._nb_ch = int(self._dataset_stats.get('nb_ch') or (int(self._dataset_stats['feature_dim']) // self._nb_mel_bins))

        if not self._is_eval:
            label_shape = self._dataset_stats.get('label_shape')
            if label_shape:
                if self._multi_accdoa is True:
                    self._num_track_dummy = int(label_shape[-3])
                    self._num_axis = int(label_shape[-2])
                    self._num_class = int(label_shape[-1])
                else:
                    self._label_len = int(label_shape[-1])
                self._doa_len = 3
            else:
                self._configure_label_shapes()

        if self._per_file:
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor(total_frames / (self._batch_size*self._feature_seq_len)))

        self._feature_batch_seq_len = self._batch_size*self._feature_seq_len
        self._label_batch_seq_len = self._batch_size*self._label_seq_len

        if self._modality == 'audio_visual':
            self._vid_feature_batch_seq_len = self._batch_size*self._vid_feature_seq_len
        return True

    def _get_filenames_list_and_feat_label_sizes(self):
        tqdm.write('Computing dataset stats')
        if self._get_filenames_list_and_feat_label_sizes_from_cache():
            return

        if self._split_selector and any(isinstance(x, str) for x in self._normalize_split_selector(self._split_selector)) and self._selected_stems is None:
            raise ValueError(
                'Split selector {} requires a valid split manifest, but none was loaded.'.format(self._split_selector)
            )

        max_frames, total_frames, temp_feat = -1, 0, []
        for filename in os.listdir(self._feat_dir):
            if self._is_selected_filename(filename):
                if self._modality == 'audio' or (hasattr(self, '_vid_feat_dir') and os.path.exists(os.path.join(self._vid_feat_dir, filename))):   # some audio files do not have corresponding videos. Ignore them.
                    self._filenames_list.append(filename)
                    temp_feat = np.load(os.path.join(self._feat_dir, filename))
                    total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len))
                    if temp_feat.shape[0]>max_frames:
                        max_frames = temp_feat.shape[0]

        if len(temp_feat)!=0:
            self._nb_frames_file = max_frames if self._per_file else temp_feat.shape[0]
            self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins
        else:
            raise ValueError('Loading features failed from {} with split selector {}'.format(self._feat_dir, self._split_selector))

        if not self._is_eval:
            self._configure_label_shapes()

        if self._per_file:
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor(total_frames / (self._batch_size*self._feature_seq_len)))

        self._feature_batch_seq_len = self._batch_size*self._feature_seq_len
        self._label_batch_seq_len = self._batch_size*self._label_seq_len

        if self._modality == 'audio_visual':
            self._vid_feature_batch_seq_len = self._batch_size*self._vid_feature_seq_len

        return

    def _prepare_per_file_batch(self, temp_feat, temp_label=None, temp_vid_feat=None):
        feat_seq_count = int(np.ceil(temp_feat.shape[0] / float(self._feature_seq_len)))
        label_seq_count = 0 if temp_label is None else int(np.ceil(temp_label.shape[0] / float(self._label_seq_len)))
        seq_count = max(feat_seq_count, label_seq_count, 1)

        target_feat_frames = seq_count * self._feature_seq_len
        if temp_feat.shape[0] < target_feat_frames:
            feat_extra_frames = target_feat_frames - temp_feat.shape[0]
            extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1]), dtype=temp_feat.dtype) * 1e-6
            temp_feat = np.concatenate((temp_feat, extra_feat), axis=0)
        else:
            temp_feat = temp_feat[:target_feat_frames]

        temp_feat = np.reshape(temp_feat, (target_feat_frames, self._nb_ch, self._nb_mel_bins))
        feat = self._split_in_seqs(temp_feat, self._feature_seq_len)
        feat = np.transpose(feat, (0, 2, 1, 3))

        vid_feat = None
        if temp_vid_feat is not None:
            target_vid_frames = seq_count * self._vid_feature_seq_len
            if temp_vid_feat.shape[0] < target_vid_frames:
                vid_extra_frames = target_vid_frames - temp_vid_feat.shape[0]
                extra_vid_feat = np.ones((vid_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2]), dtype=temp_vid_feat.dtype) * 1e-6
                temp_vid_feat = np.concatenate((temp_vid_feat, extra_vid_feat), axis=0)
            else:
                temp_vid_feat = temp_vid_feat[:target_vid_frames]
            vid_feat = self._vid_feat_split_in_seqs(temp_vid_feat, self._vid_feature_seq_len)

        label = None
        if temp_label is not None:
            target_label_frames = seq_count * self._label_seq_len
            if temp_label.shape[0] < target_label_frames:
                label_extra_frames = target_label_frames - temp_label.shape[0]
                if self._multi_accdoa is True:
                    extra_labels = np.zeros((label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class), dtype=temp_label.dtype)
                else:
                    extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]), dtype=temp_label.dtype)
                temp_label = np.concatenate((temp_label, extra_labels), axis=0)
            else:
                temp_label = temp_label[:target_label_frames]
            label = self._split_in_seqs(temp_label, self._label_seq_len)

        return feat, vid_feat, label

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()

        if self._modality == 'audio_visual':
            self._circ_buf_vid_feat = deque()

        if self._per_file:
            for filename in self._filenames_list:
                temp_feat = np.load(os.path.join(self._feat_dir, filename))
                temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, filename)) if self._modality == 'audio_visual' else None

                if self._is_eval:
                    feat, vid_feat, _ = self._prepare_per_file_batch(temp_feat, temp_vid_feat=temp_vid_feat)
                    if self._modality == 'audio_visual':
                        yield feat, vid_feat
                    else:
                        yield feat
                    continue

                temp_label = np.load(os.path.join(self._label_dir, filename))
                feat, vid_feat, label = self._prepare_per_file_batch(temp_feat, temp_label=temp_label, temp_vid_feat=temp_vid_feat)
                if self._multi_accdoa is not True:
                    mask = label[:, :, :self._nb_classes]
                    mask = np.tile(mask, 4)
                    label = mask * label[:, :, self._nb_classes:]
                if self._modality == 'audio_visual':
                    yield feat, vid_feat, label
                else:
                    yield feat, label
            return

        file_cnt = 0
        if self._is_eval:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while (len(self._circ_buf_feat) < self._feature_batch_seq_len or (hasattr(self, '_circ_buf_vid_feat') and hasattr(self, '_vid_feature_batch_seq_len') and len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len)):
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)

                    if self._modality == 'audio_visual':
                        temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, self._filenames_list[file_cnt]))
                        for vf_row_cnt, vf_row in enumerate(temp_vid_feat):
                            self._circ_buf_vid_feat.append(vf_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                        for row_cnt, row in enumerate(extra_feat):
                            self._circ_buf_feat.append(row)

                        if self._modality == 'audio_visual':
                            vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                            extra_vid_feat = np.ones((vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6

                            for vf_row_cnt, vf_row in enumerate(extra_vid_feat):
                                self._circ_buf_vid_feat.append(vf_row)

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))

                if self._modality == 'audio_visual':
                    vid_feat = np.zeros((self._vid_feature_batch_seq_len, 7, 7))
                    for v in range(self._vid_feature_batch_seq_len):
                        vid_feat[v, :, :] = self._circ_buf_vid_feat.popleft()
                    vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)

                    yield feat, vid_feat
                else:
                    yield feat

        else:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while (len(self._circ_buf_feat) < self._feature_batch_seq_len or (hasattr(self, '_circ_buf_vid_feat') and hasattr(self, '_vid_feature_batch_seq_len') and len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len)):
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))
                    if self._modality == 'audio_visual':
                        temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, self._filenames_list[file_cnt]))

                    if not self._per_file:
                        # Inorder to support variable length features, and labels of different resolution.
                        # We remove all frames in features and labels matrix that are outside
                        # the multiple of self._label_seq_len and self._feature_seq_len. Further we do this only in training.
                        temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                        temp_mul = temp_label.shape[0] // self._label_seq_len
                        temp_feat = temp_feat[:temp_mul * self._feature_seq_len, :]
                        if self._modality == 'audio_visual':
                            temp_vid_feat = temp_vid_feat[:temp_mul * self._vid_feature_seq_len, :, :]

                    for f_row in temp_feat:
                        self._circ_buf_feat.append(f_row)
                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row)

                    if self._modality == 'audio_visual':
                        for vf_row in temp_vid_feat:
                            self._circ_buf_vid_feat.append(vf_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                        if self._modality == 'audio_visual':
                            vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                            extra_vid_feat = np.ones(
                                (vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6

                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        if self._multi_accdoa is True:
                            extra_labels = np.zeros(
                                (label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                        else:
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                        for f_row in extra_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)
                        if self._modality == 'audio_visual':
                            for vf_row in extra_vid_feat:
                                self._circ_buf_vid_feat.append(vf_row)

                    file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                if self._modality == 'audio_visual':
                    vid_feat = np.zeros((self._vid_feature_batch_seq_len, 7, 7))
                    for v in range(self._vid_feature_batch_seq_len):
                        vid_feat[v, :, :] = self._circ_buf_vid_feat.popleft()

                if self._multi_accdoa is True:
                    label = np.zeros(
                        (self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                    for j in range(self._label_batch_seq_len):
                        label[j, :, :, :] = self._circ_buf_label.popleft()
                else:
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()

                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))
                if self._modality == 'audio_visual':
                    vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)

                label = self._split_in_seqs(label, self._label_seq_len)
                if self._multi_accdoa is True:
                    pass
                else:
                    mask = label[:, :, :self._nb_classes]
                    mask = np.tile(mask, 4)
                    label = mask * label[:, :, self._nb_classes:]
                if self._modality == 'audio_visual':
                    yield feat, vid_feat, label
                else:
                    yield feat, label

    def _split_in_seqs(self, data, _seq_len): # data - 250*8, 7, 64 - 250
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    def _vid_feat_split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            else:
                data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions for video features: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()
    
    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)
