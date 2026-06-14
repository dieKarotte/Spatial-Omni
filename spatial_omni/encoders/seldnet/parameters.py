# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.

import json
import os


def _load_class_count(class_mapping_path):
    if class_mapping_path and os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as handle:
            return int(json.load(handle)['class_count'])
    return None


def _is_primary_process():
    return int(os.environ.get('RANK', '0')) == 0


def get_params(argv='1'):
    log = print if _is_primary_process() else (lambda *args, **kwargs: None)
    log("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=True,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',
        load_backbone_only=False,
        pretrained_backbone_weights=None,

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='./STARSS23',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='./seld_feat_label/',

        model_dir='models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        modality='audio',  # 'audio' or 'audio_visual'
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.
        multi_gpu=False,
        gpu_ids=None,
        detect_anomaly=False,
        ddp_backend=None,
        ddp_eval=False,
        ddp_broadcast_buffers=False,
        ddp_timeout_minutes=180,
        backbone_init_mode='strict',
        structured_head_init=False,

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        batch_size=32,              # Batch size
        num_workers=max(1, min(8, os.cpu_count() or 1)),
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes
        nb_epochs=250,  # Train for maximum epochs
        # lr=1e-3,
        lr=3e-4,

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses frame-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics
        adpit_pos_weight=1.0,
        adpit_neg_weight=1.0,
        adpit_dynamic_weight=False,
        adpit_dynamic_pos_cap=1.0,
        activity_aux_weight=0.0,
        activity_aux_pos_margin=0.5,
        activity_aux_neg_margin=0.1,
        distance_loss_weight=1.0,
        explicit_sed_head=False,
        sed_pos_weight=2.0,
        sed_neg_weight=1.0,
        sed_dynamic_weight=True,
        sed_dynamic_pos_cap=20.0,
        sed_warmup_epochs=0,
        sed_warmup_weight=1.0,
        sed_main_weight=1.0,
        regression_warmup_weight=1.0,
        regression_main_weight=1.0,
        distance_warmup_scale=1.0,
        distance_main_scale=1.0,
        sed_inference_threshold=0.5,
        track_inference_threshold=0.5,
        val_subset_ratio=1.0,
        full_val_interval=1,
        val_subset_seed=1337,
        unique_classes=None,
        class_mapping_path=None,
        split_strategy='fold',
        split_manifest_path=None,
    )

    # ########### User defined parameters ##############
    if argv == '1':
        log("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        log("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        log("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        log("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        log("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        log("MIC + GCC + multi ACCDOA\n")
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        log("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '23':
        log("STARSS23 FOA + multi ACCDOA (train from scratch)\n")
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = './STARSS23'
        params['feat_label_dir'] = './seld_feat_label/starss23_foa_baseline'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['num_workers'] = max(1, min(16, os.cpu_count() or 1))

    elif argv == '230':
        log("MERGED 16k FOA + multi ACCDOA + 37 classes\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/starss23_plus_foa_16k'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['fs'] = 16000
        params['num_workers'] = max(1, min(16, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1

    elif argv == '231':
        log("MERGED 16k FOA + explicit SED head warmup + multi ACCDOA + 37 classes\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/starss23_plus_foa_16k'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['fs'] = 16000
        params['num_workers'] = max(1, min(32, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '232':
        log("MERGED 16k FOA + explicit SED head warmup + multi ACCDOA + 32 classes\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus_32cls')
        base_manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus', 'split_manifest.json')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/starss23_plus_foa_16k_32cls'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['fs'] = 16000
        params['num_workers'] = max(1, min(16, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = base_manifest
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '233':
        log("MERGED 16k FOA + explicit SED head warmup + multi ACCDOA + 29 classes\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus_29cls')
        base_manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus', 'split_manifest.json')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/starss23_plus_foa_16k_29cls'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['fs'] = 16000
        params['num_workers'] = max(1, min(16, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = base_manifest
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '234':
        log("MERGED 16k FOA + explicit SED head warmup + multi ACCDOA + 27 classes\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus_27cls')
        base_manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus', 'split_manifest.json')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/starss23_plus_foa_16k_27cls'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['fs'] = 16000
        params['num_workers'] = max(1, min(16, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = base_manifest
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '235':
        log("MERGED 16k FOA + Qwen-audio mel frontend + explicit SED head warmup + multi ACCDOA + 29 classes\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus_29cls')
        base_manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'starss23_foa_plus', 'split_manifest.json')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['load_backbone_only'] = False
        params['pretrained_backbone_weights'] = None
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/starss23_plus_foa_16k_29cls_qwenmel128'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['fs'] = 16000
        params['hop_len_s'] = 0.01
        params['label_hop_len_s'] = 0.1
        params['nb_mel_bins'] = 128
        params['hop_len'] = 160
        params['win_len'] = 400
        params['n_fft'] = 400
        params['mel_fmin'] = 0.0
        params['mel_fmax'] = 8000.0
        params['num_workers'] = max(1, min(16, os.cpu_count() or 1))
        params['batch_size'] = 16
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = base_manifest
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '236':
        log("OV simulated 16k FOA + explicit SED head warmup + multi ACCDOA + expanded 51 classes\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_plus_unmapped')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_plus_unmapped_16k_51cls'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['multi_gpu'] = True
        params['fs'] = 16000
        params['batch_size'] = 256
        params['num_workers'] = max(1, min(16, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '237':
        log("OV simulated 16k FOA + stronger 51-class head + explicit SED head warmup\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_plus_unmapped')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_plus_unmapped_16k_51cls'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['multi_gpu'] = True
        params['fs'] = 16000
        params['batch_size'] = 192
        params['num_workers'] = max(1, min(48, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['backbone_init_mode'] = 'inflate'
        params['structured_head_init'] = True
        params['val_subset_ratio'] = 0.05
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['dropout_rate'] = 0.1
        params['nb_cnn2d_filt'] = 96
        params['rnn_size'] = 256
        params['nb_fnn_layers'] = 2
        params['fnn_size'] = 256
        params['lr'] = 2e-4
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 10
        params['sed_warmup_weight'] = 12.0
        params['sed_main_weight'] = 2.0
        params['regression_warmup_weight'] = 0.1
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.45

    elif argv == '238':
        log("OV simulated 16k FOA + lax DCASE-29 labels + explicit SED head warmup\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_lax')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_lax_16k_29cls'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['multi_gpu'] = True
        params['fs'] = 16000
        params['batch_size'] = 256
        params['num_workers'] = max(1, min(48, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.35

    elif argv == '239':
        log("OV simulated 16k FOA + 51 classes + pretrained backbone + stronger head\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_plus_unmapped')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_plus_unmapped_16k_51cls'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['multi_gpu'] = True
        params['fs'] = 16000
        params['batch_size'] = 224
        params['num_workers'] = max(1, min(48, os.cpu_count() or 1))
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['backbone_init_mode'] = 'strict'
        params['structured_head_init'] = True
        params['val_subset_ratio'] = 0.05
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['dropout_rate'] = 0.1
        params['nb_fnn_layers'] = 2
        params['fnn_size'] = 256
        params['lr'] = 2e-4
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 10
        params['sed_warmup_weight'] = 12.0
        params['sed_main_weight'] = 2.0
        params['regression_warmup_weight'] = 0.1
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.45

    elif argv == '240':
        log("OV simulated 16k FOA lax DCASE-29 + Qwen-audio mel frontend for 235 checkpoint evaluation\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_lax')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['load_backbone_only'] = False
        params['pretrained_backbone_weights'] = None
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_lax_16k_29cls_qwenmel128'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['multi_gpu'] = False
        params['fs'] = 16000
        params['hop_len_s'] = 0.01
        params['label_hop_len_s'] = 0.1
        params['nb_mel_bins'] = 128
        params['hop_len'] = 160
        params['win_len'] = 400
        params['n_fft'] = 400
        params['mel_fmin'] = 0.0
        params['mel_fmax'] = 8000.0
        params['num_workers'] = max(1, min(48, os.cpu_count() or 1))
        params['batch_size'] = 16
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '241':
        log("OV simulated 16k FOA lax DCASE-29 + Qwen-audio mel frontend + sim-train normalization\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_lax')
        params['quick_test'] = False
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = 'models_audio/235_qwenmel235_run01_dev_split0_multiaccdoa_foa_best_full_model.h5'
        params['load_backbone_only'] = False
        params['pretrained_backbone_weights'] = None
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_lax_16k_29cls_qwenmel128_simnorm'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['multi_gpu'] = True
        params['fs'] = 16000
        params['hop_len_s'] = 0.01
        params['label_hop_len_s'] = 0.1
        params['nb_mel_bins'] = 128
        params['hop_len'] = 160
        params['win_len'] = 400
        params['n_fft'] = 400
        params['mel_fmin'] = 0.0
        params['mel_fmax'] = 8000.0
        params['num_workers'] = max(1, min(48, os.cpu_count() or 1))
        params['batch_size'] = 64
        params['lr'] = 1e-5
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '242':
        log("OV simulated 16k FOA lax DCASE-29 + Qwen-audio mel frontend + sim-train normalization + backbone init from task 23\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_lax')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = 'models_audio/23_starss23_foa_baseline_dev_split0_multiaccdoa_foa_model.h5'
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_lax_16k_29cls_qwenmel128_simnorm'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['multi_gpu'] = True
        params['fs'] = 16000
        params['hop_len_s'] = 0.01
        params['label_hop_len_s'] = 0.1
        params['nb_mel_bins'] = 128
        params['hop_len'] = 160
        params['win_len'] = 400
        params['n_fft'] = 400
        params['mel_fmin'] = 0.0
        params['mel_fmax'] = 8000.0
        params['num_workers'] = max(1, min(48, os.cpu_count() or 1))
        params['batch_size'] = 64
        params['lr'] = 5e-5
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['norm_fit_splits'] = ['train']
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '243':
        log("OV simulated 16k FOA lax DCASE-29 + FOA SALSA-Lite-64 + backbone init from task 23\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_lax')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = 'models_audio/23_starss23_foa_baseline_dev_split0_multiaccdoa_foa_model.h5'
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_lax_16k_29cls_foa_salsalite64'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = True
        params['foa_channel_order'] = 'WYZX'
        params['salsalite_nb_bins'] = 64
        params['multi_gpu'] = True
        params['fs'] = 16000
        params['hop_len_s'] = 0.01
        params['label_hop_len_s'] = 0.1
        params['nb_mel_bins'] = 64
        params['hop_len'] = 160
        params['win_len'] = 400
        params['n_fft'] = 400
        params['fmin_doa_salsalite'] = 50
        params['fmax_doa_salsalite'] = 2000
        params['fmax_spectra_salsalite'] = 2600
        params['num_workers'] = max(1, min(48, os.cpu_count() or 1))
        params['batch_size'] = 64
        params['lr'] = 5e-5
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['norm_fit_splits'] = ['train']
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.0
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 25.0
        params['sed_warmup_epochs'] = 5
        params['sed_warmup_weight'] = 8.0
        params['sed_main_weight'] = 1.0
        params['regression_warmup_weight'] = 0.2
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.5
        params['track_inference_threshold'] = 0.3

    elif argv == '244':
        log("OV simulated 16k FOA lax DCASE-29 + Qwen-audio mel frontend + sim-train normalization + backbone init from task 23 + SED-heavy warmup\n")
        merged_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepared_datasets', 'ovsim_foa_dcase29_lax')
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['load_backbone_only'] = True
        params['pretrained_backbone_weights'] = 'models_audio/23_starss23_foa_baseline_dev_split0_multiaccdoa_foa_model.h5'
        params['backbone_init_mode'] = 'strict'
        params['structured_head_init'] = True
        params['dataset_dir'] = merged_root
        params['feat_label_dir'] = './seld_feat_label/ovsim_foa_dcase29_lax_16k_29cls_qwenmel128_simnorm'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['multi_gpu'] = True
        params['fs'] = 16000
        params['hop_len_s'] = 0.01
        params['label_hop_len_s'] = 0.1
        params['nb_mel_bins'] = 128
        params['hop_len'] = 160
        params['win_len'] = 400
        params['n_fft'] = 400
        params['mel_fmin'] = 0.0
        params['mel_fmax'] = 8000.0
        params['num_workers'] = max(1, min(48, os.cpu_count() or 1))
        params['batch_size'] = 64
        params['lr'] = 5e-5
        params['dropout_rate'] = 0.1
        params['nb_fnn_layers'] = 2
        params['fnn_size'] = 256
        params['split_strategy'] = 'manifest'
        params['split_manifest_path'] = os.path.join(merged_root, 'split_manifest.json')
        params['norm_fit_splits'] = ['train']
        params['class_mapping_path'] = os.path.join(merged_root, 'class_mapping.json')
        params['val_subset_ratio'] = 0.1
        params['full_val_interval'] = 10
        params['explicit_sed_head'] = True
        params['adpit_pos_weight'] = 1.0
        params['adpit_neg_weight'] = 0.01
        params['adpit_dynamic_weight'] = True
        params['adpit_dynamic_pos_cap'] = 20.0
        params['activity_aux_weight'] = 1.0
        params['activity_aux_pos_margin'] = 0.5
        params['activity_aux_neg_margin'] = 0.02
        params['distance_loss_weight'] = 0.1
        params['sed_pos_weight'] = 2.5
        params['sed_neg_weight'] = 1.0
        params['sed_dynamic_weight'] = True
        params['sed_dynamic_pos_cap'] = 30.0
        params['sed_warmup_epochs'] = 8
        params['sed_warmup_weight'] = 10.0
        params['sed_main_weight'] = 1.5
        params['regression_warmup_weight'] = 0.1
        params['regression_main_weight'] = 1.0
        params['distance_warmup_scale'] = 0.0
        params['distance_main_scale'] = 1.0
        params['sed_inference_threshold'] = 0.45
        params['track_inference_threshold'] = 0.3

    elif argv == '24':
        log("STARSS23 MIC + GCC + multi ACCDOA (train from scratch)\n")
        params['quick_test'] = False
        params['finetune_mode'] = False
        params['dataset_dir'] = './STARSS23'
        params['feat_label_dir'] = './seld_feat_label/starss23_mic_baseline'
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['num_workers'] = max(1, min(16, os.cpu_count() or 1))

    elif argv == '999':
        log("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        log('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling
    # params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached
    params['patience'] = 50 
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']

    params['unique_classes'] = _load_class_count(params['class_mapping_path']) or params['unique_classes']

    if params['unique_classes'] is None and '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif params['unique_classes'] is None and '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif params['unique_classes'] is None and '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif params['unique_classes'] is None and ('2023' in params['dataset_dir'] or 'STARSS23' in params['dataset_dir']):
        params['unique_classes'] = 13
    elif params['unique_classes'] is None and '2024' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        log("\t{}: {}".format(key, value))
    return params
