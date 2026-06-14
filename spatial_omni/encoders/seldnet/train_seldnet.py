#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
from contextlib import nullcontext
from datetime import timedelta
import json
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import parameters
import time
import math
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model 

def split_model_output(output):
    if isinstance(output, dict):
        return output['accdoa'], output.get('sed_logits')
    return output, None


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5
      
    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes, sed_logits=None, sed_threshold=0.5, track_threshold=0.5):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    dist0 = accdoa_in[:, :, 3*nb_classes:4*nb_classes]
    dist0[dist0 < 0.] = 0.
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > track_threshold
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes], accdoa_in[:, :, 6*nb_classes:7*nb_classes]
    dist1 = accdoa_in[:, :, 7*nb_classes:8*nb_classes]
    dist1[dist1<0.] = 0.
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > track_threshold
    doa1 = accdoa_in[:, :, 4*nb_classes: 7*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 8*nb_classes:9*nb_classes], accdoa_in[:, :, 9*nb_classes:10*nb_classes], accdoa_in[:, :, 10*nb_classes:11*nb_classes]
    dist2 = accdoa_in[:, :, 11*nb_classes:]
    dist2[dist2<0.] = 0.
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > track_threshold
    doa2 = accdoa_in[:, :, 8*nb_classes:11*nb_classes]

    if sed_logits is not None:
        sed_prob = 1.0 / (1.0 + np.exp(-sed_logits))
        sed_gate = sed_prob > sed_threshold
        sed0 = sed0 & sed_gate
        sed1 = sed1 & sed_gate
        sed2 = sed2 & sed_gate

    return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def sample_validation_subset(file_list, subset_ratio, seed, epoch_cnt):
    if subset_ratio >= 1.0 or len(file_list) <= 1:
        return list(file_list)
    subset_size = max(1, int(math.ceil(len(file_list) * subset_ratio)))
    rng = np.random.default_rng(seed + epoch_cnt)
    subset_indices = np.sort(rng.choice(len(file_list), size=subset_size, replace=False))
    return [file_list[idx] for idx in subset_indices]


def unwrap_model(model):
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and 'state_dict' in checkpoint_obj and isinstance(checkpoint_obj['state_dict'], dict):
        return checkpoint_obj['state_dict']
    return checkpoint_obj


def _strip_module_prefix(state_dict):
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint_state_dict(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _extract_state_dict(state_dict)
    return _strip_module_prefix(state_dict)


def load_model_state(model, checkpoint_path, strict=True):
    base_model = unwrap_model(model)
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    return base_model.load_state_dict(state_dict, strict=strict)


def save_model_state(model, checkpoint_path):
    torch.save(unwrap_model(model).state_dict(), checkpoint_path)


def is_distributed_launch():
    return int(os.environ.get('WORLD_SIZE', '1')) > 1


def is_main_process(dist_ctx):
    return dist_ctx['rank'] == 0


def dist_tqdm_write(dist_ctx, message):
    if is_main_process(dist_ctx):
        tqdm.write(message)


def dist_barrier(dist_ctx):
    if dist_ctx['enabled'] and dist.is_initialized():
        if dist_ctx.get('backend') == 'nccl' and dist_ctx.get('device') is not None and dist_ctx['device'].type == 'cuda':
            dist.barrier(device_ids=[dist_ctx['local_rank']])
        else:
            dist.barrier()


def cleanup_distributed(dist_ctx):
    if dist_ctx['enabled'] and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device_and_gpu_ids(params):
    if not torch.cuda.is_available():
        return torch.device('cpu'), []

    available_gpu_ids = list(range(torch.cuda.device_count()))
    requested_gpu_ids = params.get('gpu_ids')
    if requested_gpu_ids is None:
        selected_gpu_ids = available_gpu_ids
    elif isinstance(requested_gpu_ids, str):
        selected_gpu_ids = [int(idx.strip()) for idx in requested_gpu_ids.split(',') if idx.strip()]
    else:
        selected_gpu_ids = [int(idx) for idx in requested_gpu_ids]

    selected_gpu_ids = [idx for idx in selected_gpu_ids if idx in available_gpu_ids]
    if not selected_gpu_ids:
        selected_gpu_ids = [available_gpu_ids[0]]

    return torch.device('cuda:{}'.format(selected_gpu_ids[0])), selected_gpu_ids


def init_distributed_context(params):
    if params.get('multi_gpu') and is_distributed_launch():
        if not dist.is_available():
            raise RuntimeError('torch.distributed is not available, but torchrun/DDP was requested')

        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        backend = params.get('ddp_backend') or ('nccl' if torch.cuda.is_available() else 'gloo')

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda:{}'.format(local_rank))
        else:
            device = torch.device('cpu')

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                timeout=timedelta(minutes=float(params.get('ddp_timeout_minutes', 180))),
            )

        return {
            'enabled': True,
            'rank': rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'device': device,
            'gpu_ids': [local_rank] if device.type == 'cuda' else [],
            'backend': backend,
        }

    device, gpu_ids = resolve_device_and_gpu_ids(params)
    return {
        'enabled': False,
        'rank': 0,
        'world_size': 1,
        'local_rank': gpu_ids[0] if gpu_ids else 0,
        'device': device,
        'gpu_ids': gpu_ids,
        'backend': None,
    }


def maybe_wrap_model_for_multi_gpu(model, params, dist_ctx):
    device = dist_ctx['device']
    gpu_ids = dist_ctx['gpu_ids']
    model = model.to(device)

    if dist_ctx['enabled']:
        dist_tqdm_write(
            dist_ctx,
            'Using torchrun + DDP (backend={}, world_size={}, local_rank={})'.format(
                dist_ctx['backend'], dist_ctx['world_size'], dist_ctx['local_rank']
            )
        )
        ddp_kwargs = {}
        if device.type == 'cuda':
            ddp_kwargs['device_ids'] = [dist_ctx['local_rank']]
            ddp_kwargs['output_device'] = dist_ctx['local_rank']
        ddp_kwargs['broadcast_buffers'] = bool(params.get('ddp_broadcast_buffers', False))
        return DDP(model, **ddp_kwargs)

    if device.type != 'cuda':
        tqdm.write('Using CPU for training/evaluation')
        return model

    if params.get('multi_gpu') and len(gpu_ids) > 1:
        tqdm.write('Using nn.DataParallel on GPUs {}'.format(gpu_ids))
        return nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    if params.get('multi_gpu') and len(gpu_ids) <= 1:
        tqdm.write('multi_gpu requested, but only one visible GPU is available; falling back to single-GPU mode')
    else:
        tqdm.write('Using single GPU {}'.format(gpu_ids[0]))
    return model


def configure_runtime_params_for_ddp(params, dist_ctx):
    if not dist_ctx['enabled']:
        return params

    params = dict(params)
    original_batch_size = int(params['batch_size'])
    params['global_batch_size'] = original_batch_size
    params['batch_size'] = max(1, original_batch_size // dist_ctx['world_size'])
    if original_batch_size % dist_ctx['world_size'] != 0:
        dist_tqdm_write(
            dist_ctx,
            'Global batch_size {} is not divisible by world_size {}; using per-rank batch_size {}'.format(
                original_batch_size, dist_ctx['world_size'], params['batch_size']
            )
        )
    else:
        dist_tqdm_write(
            dist_ctx,
            'DDP batch sizing: global batch_size {} -> per-rank batch_size {}'.format(
                original_batch_size, params['batch_size']
            )
        )
    return params


def shard_file_list(file_list, dist_ctx):
    if not dist_ctx['enabled'] or len(file_list) <= 1:
        return list(file_list)
    return list(file_list[dist_ctx['rank']::dist_ctx['world_size']])


def estimate_file_workloads(file_list, data_generator, per_file):
    stats = getattr(data_generator, '_dataset_stats', None) or {}
    files_meta = stats.get('files', {})
    seq_len = int(getattr(data_generator, '_feature_seq_len', 1))
    workloads = {}
    for filename in file_list:
        feat_frames = int(files_meta.get(filename, {}).get('feat_frames', seq_len))
        if per_file:
            workloads[filename] = max(1, int(math.ceil(float(feat_frames) / max(1, seq_len))))
        else:
            usable_frames = feat_frames - (feat_frames % max(1, seq_len))
            workloads[filename] = max(1, usable_frames)
    return workloads


def balance_file_list(file_list, data_generator, dist_ctx, per_file=False):
    if not dist_ctx['enabled'] or len(file_list) <= 1:
        return list(file_list)

    workloads = estimate_file_workloads(file_list, data_generator, per_file=per_file)
    rank_assignments = [[] for _ in range(dist_ctx['world_size'])]
    rank_costs = [0 for _ in range(dist_ctx['world_size'])]

    for filename in sorted(file_list, key=lambda item: (workloads.get(item, 1), item), reverse=True):
        target_rank = min(range(dist_ctx['world_size']), key=lambda idx: (rank_costs[idx], len(rank_assignments[idx])))
        rank_assignments[target_rank].append(filename)
        rank_costs[target_rank] += workloads.get(filename, 1)

    return sorted(rank_assignments[dist_ctx['rank']])


def create_data_generator(params, split, dist_ctx, shuffle=True, per_file=False, is_eval=False, selected_files=None, shard=False):
    base_gen = cls_data_generator.DataGenerator(
        params=params,
        split=split,
        shuffle=shuffle,
        per_file=per_file,
        is_eval=is_eval,
        selected_files=selected_files,
    )
    full_filelist = list(base_gen.get_filelist())
    if not shard or not dist_ctx['enabled']:
        return base_gen, full_filelist, full_filelist

    local_files = balance_file_list(full_filelist, base_gen, dist_ctx, per_file=per_file)
    if not local_files:
        raise ValueError('Rank {} received no files for split {}'.format(dist_ctx['rank'], split))

    sharded_gen = cls_data_generator.DataGenerator(
        params=params,
        split=split,
        shuffle=shuffle,
        per_file=per_file,
        is_eval=is_eval,
        selected_files=local_files,
    )
    return sharded_gen, full_filelist, local_files


def reduce_loss_stats(loss_sum, batch_count, device, dist_ctx):
    if not dist_ctx['enabled']:
        return float(loss_sum), int(batch_count)

    stats_tensor = torch.tensor([loss_sum, batch_count], device=device, dtype=torch.float64)
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    return float(stats_tensor[0].item()), int(stats_tensor[1].item())


def broadcast_stop_flag(stop_flag, device, dist_ctx):
    if not dist_ctx['enabled']:
        return bool(stop_flag)

    flag_tensor = torch.tensor([1 if stop_flag else 0], device=device, dtype=torch.int64)
    dist.broadcast(flag_tensor, src=0)
    return bool(flag_tensor.item())


def init_run_sync_dir(base_path):
    os.makedirs(base_path, exist_ok=True)


def sync_signal_path(sync_dir, name):
    return os.path.join(sync_dir, '{}.json'.format(name))


def remove_sync_signal(path):
    if os.path.exists(path):
        os.remove(path)


def write_sync_signal(path, payload):
    tmp_path = '{}.tmp'.format(path)
    with open(tmp_path, 'w') as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)


def wait_for_sync_signal(path, poll_interval_s=5.0):
    while True:
        if os.path.exists(path):
            with open(path, 'r') as handle:
                return json.load(handle)
        time.sleep(poll_interval_s)


def _repeat_tensor_to_shape(source_tensor, target_shape):
    expanded = source_tensor
    for dim, target_size in enumerate(target_shape):
        if expanded.shape[dim] >= target_size:
            continue
        repeat_factors = [1] * expanded.dim()
        repeat_factors[dim] = int(math.ceil(float(target_size) / float(expanded.shape[dim])))
        expanded = expanded.repeat(*repeat_factors)
    slices = tuple(slice(0, target_size) for target_size in target_shape)
    return expanded[slices].clone()


def _inflate_grouped_first_dim(source_tensor, target_shape, groups):
    if source_tensor.shape[0] % groups != 0 or target_shape[0] % groups != 0:
        return None
    source_chunk = source_tensor.shape[0] // groups
    target_chunk = target_shape[0] // groups
    inflated_chunks = []
    for group_idx in range(groups):
        group_source = source_tensor[group_idx * source_chunk:(group_idx + 1) * source_chunk]
        group_target_shape = (target_chunk,) + tuple(target_shape[1:])
        inflated_chunks.append(_repeat_tensor_to_shape(group_source, group_target_shape))
    return torch.cat(inflated_chunks, dim=0)


def _inflate_pretrained_tensor(key, source_tensor, target_tensor):
    if source_tensor.dim() != target_tensor.dim():
        return None
    if any(source_dim > target_dim for source_dim, target_dim in zip(source_tensor.shape, target_tensor.shape)):
        return None

    if key.startswith('gru.') and any(tag in key for tag in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')):
        inflated = _inflate_grouped_first_dim(source_tensor, target_tensor.shape, groups=3)
        if inflated is not None:
            return inflated.to(dtype=target_tensor.dtype)

    if 'mhsa_block_list.' in key and ('in_proj_weight' in key or 'in_proj_bias' in key):
        inflated = _inflate_grouped_first_dim(source_tensor, target_tensor.shape, groups=3)
        if inflated is not None:
            return inflated.to(dtype=target_tensor.dtype)

    return _repeat_tensor_to_shape(source_tensor, target_tensor.shape).to(dtype=target_tensor.dtype)


def initialize_structured_heads(model, params):
    if not params.get('structured_head_init'):
        return []

    base_model = unwrap_model(model)
    initialized_layers = []
    with torch.no_grad():
        if hasattr(base_model, 'fnn_list') and len(base_model.fnn_list):
            hidden_layers = list(base_model.fnn_list[:-1])
            for layer_idx, layer in enumerate(hidden_layers):
                if not isinstance(layer, nn.Linear):
                    continue
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
                diag_size = min(layer.out_features, layer.in_features)
                layer.weight[:diag_size, :diag_size] = torch.eye(
                    diag_size, device=layer.weight.device, dtype=layer.weight.dtype
                )
                initialized_layers.append('fnn_list.{}'.format(layer_idx))

            final_layer = base_model.fnn_list[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.xavier_uniform_(final_layer.weight)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)
                initialized_layers.append('fnn_list.{}'.format(len(base_model.fnn_list) - 1))

        if getattr(base_model, 'use_explicit_sed_head', False) and hasattr(base_model, 'sed_head'):
            nn.init.xavier_uniform_(base_model.sed_head.weight)
            if base_model.sed_head.bias is not None:
                nn.init.zeros_(base_model.sed_head.bias)
            initialized_layers.append('sed_head')

    return initialized_layers


def load_backbone_weights(model, weights_path, init_mode='strict'):
    if not os.path.exists(weights_path):
        raise FileNotFoundError('Backbone weights not found: {}'.format(weights_path))

    prefixes = ('conv_block_list.', 'gru.', 'mhsa_block_list.', 'layer_norm_list.')
    state_dict = load_checkpoint_state_dict(weights_path)
    base_model = unwrap_model(model)
    model_state = base_model.state_dict()

    filtered_state = {}
    loaded_exact = []
    loaded_inflated = []
    skipped = []
    for key, value in state_dict.items():
        if not key.startswith(prefixes) or key not in model_state:
            continue

        target_value = model_state[key]
        if target_value.shape == value.shape:
            filtered_state[key] = value.to(dtype=target_value.dtype)
            loaded_exact.append(key)
            continue

        if init_mode == 'inflate':
            inflated_value = _inflate_pretrained_tensor(key, value, target_value)
            if inflated_value is not None:
                filtered_state[key] = inflated_value
                loaded_inflated.append(key)
                continue

        skipped.append(key)

    missing, unexpected = base_model.load_state_dict(filtered_state, strict=False)
    return loaded_exact, loaded_inflated, skipped, missing, unexpected


def eval_epoch(data_generator, model, dcase_output_folder, params, device, dist_ctx=None, desc='Eval'):
    eval_filelist = data_generator.get_filelist()
    model.eval()
    file_cnt = 0
    total_batches = data_generator.get_total_batches_in_data()
    progress_disabled = dist_ctx is not None and not is_main_process(dist_ctx)
    with torch.no_grad(), tqdm(total=total_batches, desc=desc, leave=False, disable=progress_disabled) as pbar:
        for values in data_generator.generate():
            if len(values) == 2: # audio visual
                data, vid_feat = values
                data, vid_feat = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float()
                output = model(data, vid_feat)
            else:
                data = values
                data = torch.tensor(data).to(device).float()
                output = model(data)
            accdoa_output, sed_logits = split_model_output(output)
            accdoa_np = accdoa_output.detach().cpu().numpy()
            sed_logits_np = sed_logits.detach().cpu().numpy() if sed_logits is not None else None

            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(
                    accdoa_np, params['unique_classes'],
                    sed_logits=sed_logits_np if params.get('explicit_sed_head') else None,
                    sed_threshold=float(params.get('sed_inference_threshold', 0.5)),
                    track_threshold=float(params.get('track_inference_threshold', 0.5))
                )
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                dist_pred0 = reshape_3Dto2D(dist_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                dist_pred1 = reshape_3Dto2D(dist_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
                dist_pred2 = reshape_3Dto2D(dist_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file

            output_file = os.path.join(dcase_output_folder, eval_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                            if sed_pred1[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                            if sed_pred2[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]])
            data_generator.write_output_format_file(output_file, output_dict)
            pbar.update(1)


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device, dist_ctx=None, desc='Validation'):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    test_filelist = data_generator.get_filelist()

    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    total_batches = data_generator.get_total_batches_in_data()
    progress_disabled = dist_ctx is not None and not is_main_process(dist_ctx)
    with torch.no_grad(), tqdm(total=total_batches, desc=desc, leave=False, disable=progress_disabled) as pbar:
        for values in data_generator.generate():
            if len(values) == 2:
                data, target = values
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
                output = model(data)
            elif len(values) == 3:
                data, vid_feat, target = values
                data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()
                output = model(data, vid_feat)
            loss = criterion(output, target)
            accdoa_output, sed_logits = split_model_output(output)
            accdoa_np = accdoa_output.detach().cpu().numpy()
            sed_logits_np = sed_logits.detach().cpu().numpy() if sed_logits is not None else None

            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(
                    accdoa_np, params['unique_classes'],
                    sed_logits=sed_logits_np if params.get('explicit_sed_head') else None,
                    sed_threshold=float(params.get('sed_inference_threshold', 0.5)),
                    track_threshold=float(params.get('track_inference_threshold', 0.5))
                )
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                dist_pred0 = reshape_3Dto2D(dist_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                dist_pred1 = reshape_3Dto2D(dist_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
                dist_pred2 = reshape_3Dto2D(dist_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file

            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                            if sed_pred1[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                            if sed_pred2[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]]) 
            data_generator.write_output_format_file(output_file, output_dict)


            test_loss += loss.item()
            nb_test_batches += 1
            pbar.set_postfix(loss='{:.4f}'.format(test_loss / nb_test_batches))
            pbar.update(1)
            if params['quick_test'] and nb_test_batches == 4:
                break

        test_loss, nb_test_batches = reduce_loss_stats(test_loss, nb_test_batches, device, dist_ctx or {'enabled': False})
        test_loss /= max(1, nb_test_batches)

    return test_loss


def train_epoch(data_generator, optimizer, model, criterion, params, device, dist_ctx=None, desc='Train'):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    total_batches = data_generator.get_total_batches_in_data()
    progress_disabled = dist_ctx is not None and not is_main_process(dist_ctx)
    join_context = model.join if isinstance(model, DDP) else nullcontext
    with join_context():
        with tqdm(total=total_batches, desc=desc, leave=False, disable=progress_disabled) as pbar:
            for values in data_generator.generate():
                # load one batch of data
                if len(values) == 2:
                    data, target = values
                    data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
                    optimizer.zero_grad()
                    output = model(data)
                elif len(values) == 3:
                    data, vid_feat, target = values
                    data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()
                    optimizer.zero_grad()
                    output = model(data, vid_feat)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                nb_train_batches += 1
                pbar.set_postfix(loss='{:.4f}'.format(train_loss / nb_train_batches))
                pbar.update(1)
                if params['quick_test'] and nb_train_batches == 4:
                    break

    train_loss, nb_train_batches = reduce_loss_stats(train_loss, nb_train_batches, device, dist_ctx or {'enabled': False})
    train_loss /= max(1, nb_train_batches)

    return train_loss


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    tqdm.write('argv: {}'.format(argv))
    if len(argv) != 3:
        tqdm.write('Usage: python train_seldnet.py <task-id> <job-id>. Falling back to defaults.')

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)
    dist_ctx = init_distributed_context(params)
    params = configure_runtime_params_for_ddp(params, dist_ctx)
    device = dist_ctx['device']
    job_id = 1 if len(argv) < 3 else argv[-1]

    torch.autograd.set_detect_anomaly(bool(params.get('detect_anomaly', False)))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    try:
        # Training setup
        train_splits, val_splits, test_splits = None, None, None

        if params['mode'] == 'dev':
            if params.get('split_strategy') == 'manifest':
                train_splits = ['train']
                val_splits = ['valid']
                test_splits = ['test']
            elif '2020' in params['dataset_dir']:
                test_splits = [1]
                val_splits = [2]
                train_splits = [[3, 4, 5, 6]]

            elif '2021' in params['dataset_dir']:
                test_splits = [6]
                val_splits = [5]
                train_splits = [[1, 2, 3, 4]]

            elif '2022' in params['dataset_dir']:
                test_splits = [[4]]
                val_splits = [[4]]
                train_splits = [[1, 2, 3]]
            elif '2023' in params['dataset_dir'] or 'STARSS23' in params['dataset_dir']:
                test_splits = [[4]]
                val_splits = [[4]]
                train_splits = [[1, 2, 3]]
            elif '2024' in params['dataset_dir']:
                test_splits = [[4]]
                val_splits = [[4]]
                train_splits = [[3]]

            else:
                dist_tqdm_write(dist_ctx, 'ERROR: Unknown dataset splits')
                exit()

            for split_cnt, split in enumerate(test_splits):
                dist_tqdm_write(dist_ctx, 'Starting split {}'.format(split))

                # Unique name for the run
                loc_feat = params['dataset']
                if params['dataset'] == 'mic':
                    if params['use_salsalite']:
                        loc_feat = '{}_salsa'.format(params['dataset'])
                    else:
                        loc_feat = '{}_gcc'.format(params['dataset'])
                loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

                if is_main_process(dist_ctx):
                    cls_feature_class.create_folder(params['model_dir'])
                    cls_feature_class.create_folder(params['dcase_output_dir'])

                unique_name = '{}_{}_{}_split{}_{}_{}'.format(
                    task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
                )
                model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
                sync_dir = model_name.replace('_model.h5', '_ddp_sync')
                if is_main_process(dist_ctx):
                    cls_feature_class.delete_and_create_folder(sync_dir)
                else:
                    init_run_sync_dir(sync_dir)
                dist_tqdm_write(dist_ctx, 'Run name: {}'.format(unique_name))

                # Load train and validation data
                dist_tqdm_write(dist_ctx, 'Loading training dataset')
                data_gen_train, train_filelist_full, train_filelist_local = create_data_generator(
                    params=params, split=train_splits[split_cnt], dist_ctx=dist_ctx, shard=dist_ctx['enabled']
                )

                dist_tqdm_write(dist_ctx, 'Loading validation dataset')
                data_gen_val, val_filelist_full, val_filelist_local = create_data_generator(
                    params=params, split=val_splits[split_cnt], dist_ctx=dist_ctx, shuffle=False, per_file=True, shard=dist_ctx['enabled']
                )

                dist_tqdm_write(
                    dist_ctx,
                    'Dataset shards: train global/local = {}/{} ; valid global/local = {}/{}'.format(
                        len(train_filelist_full), len(train_filelist_local), len(val_filelist_full), len(val_filelist_local)
                    )
                )

                # Collect i/o data size and load model configuration
                if params['modality'] == 'audio_visual':
                    data_in, vid_data_in, data_out = data_gen_train.get_data_sizes()
                    base_model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in)
                else:
                    data_in, data_out = data_gen_train.get_data_sizes()
                    base_model = seldnet_model.SeldModel(data_in, data_out, params)

                if params.get('load_backbone_only'):
                    dist_tqdm_write(dist_ctx, 'Loading backbone weights from {}'.format(params['pretrained_backbone_weights']))
                    loaded_keys, inflated_keys, skipped_keys, missing_keys, unexpected_keys = load_backbone_weights(
                        base_model,
                        params['pretrained_backbone_weights'],
                        init_mode=params.get('backbone_init_mode', 'strict')
                    )
                    dist_tqdm_write(dist_ctx, 'Backbone load: exact={} inflated={} skipped={} missing={} unexpected={}'.format(
                        len(list(loaded_keys)), len(list(inflated_keys)), len(skipped_keys), len(missing_keys), len(unexpected_keys)
                    ))
                elif params['finetune_mode']:
                    dist_tqdm_write(dist_ctx, 'Finetuning from {}'.format(params['pretrained_model_weights']))
                    state_dict = load_checkpoint_state_dict(params['pretrained_model_weights'])
                    if params['modality'] == 'audio_visual':
                        state_dict = {k: v for k, v in state_dict.items() if 'fnn' not in k}
                    base_model.load_state_dict(state_dict, strict=False)

                initialized_head_layers = initialize_structured_heads(base_model, params)
                if initialized_head_layers:
                    dist_tqdm_write(dist_ctx, 'Structured head init: {}'.format(', '.join(initialized_head_layers)))

                model = maybe_wrap_model_for_multi_gpu(base_model, params, dist_ctx)

                dist_tqdm_write(dist_ctx, 'SELD-net data_in={} data_out={}'.format(data_in, data_out))
                dist_tqdm_write(dist_ctx, 'Model cfg: dropout_rate={} nb_cnn_filt={} f_pool_size={} t_pool_size={} rnn_size={} nb_attention_blocks={} fnn_size={}'.format(
                    params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'], params['nb_self_attn_layers'],
                    params['fnn_size']))
                if is_main_process(dist_ctx):
                    tqdm.write(str(base_model))
                    if model is not base_model:
                        tqdm.write('Model wrapper: {}'.format(model.__class__.__name__))

                # Dump results in DCASE output format for calculating final scores
                dcase_output_val_subset_folder = os.path.join(params['dcase_output_dir'], '{}_val_subset'.format(unique_name))
                dcase_output_val_full_folder = os.path.join(params['dcase_output_dir'], '{}_val_full'.format(unique_name))
                dist_tqdm_write(dist_ctx, 'Validation predictions (subset) -> {}'.format(dcase_output_val_subset_folder))
                dist_tqdm_write(dist_ctx, 'Validation predictions (full) -> {}'.format(dcase_output_val_full_folder))

                # Initialize evaluation metric class lazily on the main process only.
                score_obj = None

                # start training
                best_val_epoch = -1
                best_val_mode = 'none'
                best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err, best_rel_dist_err = 1., 0., 180., 0., 9999., 999999., 999999.
                best_full_epoch = -1
                best_full_ER, best_full_F, best_full_LE, best_full_LR, best_full_seld_scr, best_full_dist_err, best_full_rel_dist_err = 1., 0., 180., 0., 9999., 999999., 999999.
                best_full_model_name = model_name.replace('_model.h5', '_best_full_model.h5')
                patience_cnt = 0

                nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
                optimizer = optim.Adam(model.parameters(), lr=params['lr'])
                if params['multi_accdoa'] is True:
                    criterion = seldnet_model.MSELoss_ADPIT(params)
                else:
                    criterion = nn.MSELoss()
                if params.get('explicit_sed_head'):
                    criterion = seldnet_model.SEDDOAMultiTaskLoss(params)

                epoch_iterator = tqdm(range(nb_epoch), desc='Epochs', leave=True, disable=not is_main_process(dist_ctx))
                for epoch_cnt in epoch_iterator:
                    if hasattr(criterion, 'set_epoch'):
                        criterion.set_epoch(epoch_cnt)
                    # ---------------------------------------------------------------------
                    # TRAINING
                    # ---------------------------------------------------------------------
                    start_time = time.time()
                    train_loss = train_epoch(
                        data_gen_train, optimizer, model, criterion, params, device, dist_ctx=dist_ctx,
                        desc='Train {:03d}'.format(epoch_cnt + 1)
                    )
                    train_time = time.time() - start_time

                    # ---------------------------------------------------------------------
                    # VALIDATION
                    # ---------------------------------------------------------------------
                    start_time = time.time()
                    run_full_val = (((epoch_cnt + 1) % max(1, int(params.get('full_val_interval', 1)))) == 0)
                    if run_full_val:
                        val_global_files = list(val_filelist_full)
                        val_output_folder = dcase_output_val_full_folder
                        val_desc = 'Val(full) {:03d}'.format(epoch_cnt + 1)
                        val_mode = 'full'
                    else:
                        val_global_files = sample_validation_subset(
                            val_filelist_full,
                            float(params.get('val_subset_ratio', 1.0)),
                            int(params.get('val_subset_seed', 1337)),
                            epoch_cnt
                        )
                        val_output_folder = dcase_output_val_subset_folder
                        val_desc = 'Val(subset) {:03d}'.format(epoch_cnt + 1)
                        val_mode = 'subset'

                    run_ddp_eval = dist_ctx['enabled'] and bool(params.get('ddp_eval', False))
                    val_local_files = balance_file_list(val_global_files, data_gen_val, dist_ctx, per_file=True) if run_ddp_eval else list(val_global_files)
                    rank0_val_local_count = len(val_local_files)
                    if run_ddp_eval and not is_main_process(dist_ctx):
                        rank0_val_local_count = len(balance_file_list(
                            val_global_files,
                            data_gen_val,
                            dict(dist_ctx, rank=0),
                            per_file=True
                        ))
                    if is_main_process(dist_ctx):
                        cls_feature_class.delete_and_create_folder(val_output_folder)
                        tqdm.write(
                            'Epoch {:03d}: validation mode={} global_files={} local_files(rank0)={} ddp_eval={}'.format(
                                epoch_cnt + 1, val_mode, len(val_global_files), rank0_val_local_count,
                                run_ddp_eval
                            )
                        )

                    folder_ready_signal = sync_signal_path(sync_dir, 'epoch_{:03d}_{}_folder_ready'.format(epoch_cnt + 1, val_mode))
                    result_signal = sync_signal_path(sync_dir, 'epoch_{:03d}_{}_result'.format(epoch_cnt + 1, val_mode))
                    if is_main_process(dist_ctx):
                        remove_sync_signal(folder_ready_signal)
                        remove_sync_signal(result_signal)
                        cls_feature_class.delete_and_create_folder(val_output_folder)
                        write_sync_signal(folder_ready_signal, {'ready': True})
                    elif run_ddp_eval:
                        wait_for_sync_signal(folder_ready_signal)

                    val_loss = None
                    if run_ddp_eval:
                        val_gen_epoch = cls_data_generator.DataGenerator(
                            params=params,
                            split=val_splits[split_cnt],
                            shuffle=False,
                            per_file=True,
                            selected_files=val_local_files
                        )
                        val_loss = test_epoch(
                            val_gen_epoch, model, criterion, val_output_folder, params, device, dist_ctx=dist_ctx, desc=val_desc
                        )
                    else:
                        if is_main_process(dist_ctx):
                            val_gen_epoch = cls_data_generator.DataGenerator(
                                params=params,
                                split=val_splits[split_cnt],
                                shuffle=False,
                                per_file=True,
                                selected_files=val_local_files
                            )
                            val_loss = test_epoch(
                                val_gen_epoch, unwrap_model(model), criterion, val_output_folder, params, device, dist_ctx=None, desc=val_desc
                            )

                    should_stop = False
                    if is_main_process(dist_ctx):
                        if score_obj is None:
                            tqdm.write(
                                'Building validation reference index from {}'.format(
                                    os.path.join(params['dataset_dir'], 'metadata_dev')
                                )
                            )
                            score_obj = ComputeSELDResults(params)
                        tqdm.write('Scoring validation outputs from {}'.format(val_output_folder))
                        val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(val_output_folder)
                        val_time = time.time() - start_time

                        # Save best checkpoint on both subset and full validation rounds using overall SELD score.
                        if val_seld_scr <= best_seld_scr:
                            best_val_epoch, best_val_mode = epoch_cnt, val_mode
                            best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err = val_ER, val_F, val_LE, val_LR, val_seld_scr, val_dist_err
                            best_rel_dist_err = val_rel_dist_err
                            save_model_state(model, model_name)

                        # Track a separate full-validation best for early stopping and a more stable reference.
                        if run_full_val:
                            if val_seld_scr <= best_full_seld_scr:
                                best_full_epoch = epoch_cnt
                                best_full_ER, best_full_F, best_full_LE, best_full_LR = val_ER, val_F, val_LE, val_LR
                                best_full_seld_scr, best_full_dist_err, best_full_rel_dist_err = val_seld_scr, val_dist_err, val_rel_dist_err
                                save_model_state(model, best_full_model_name)
                                patience_cnt = 0
                            else:
                                patience_cnt += 1

                        val_ang_acc = max(0.0, 1.0 - (val_LE / 180.0)) if np.isfinite(val_LE) else np.nan
                        best_ang_acc = max(0.0, 1.0 - (best_LE / 180.0)) if np.isfinite(best_LE) else np.nan
                        best_full_ang_acc = max(0.0, 1.0 - (best_full_LE / 180.0)) if np.isfinite(best_full_LE) else np.nan

                        epoch_summary = (
                            'epoch={:03d} val_mode={} train_time={:.2f}s val_time={:.2f}s '
                            'train_loss={:.4f} val_loss={:.4f} '
                            'ER/F/LR={:.3f}/{:.3f}/{:.3f} '
                            'AngE_deg/AngAcc={:.2f}/{:.3f} '
                            'Dist/RelDist/SELD={:.2f}/{:.2f}/{:.2f} '
                            'best_any(epoch={},mode={})='
                            'ER:{:.3f} F:{:.3f} LR:{:.3f} AngE:{:.2f} AngAcc:{:.3f} Dist:{:.2f} RelDist:{:.2f} SELD:{:.2f} '
                            'best_full(epoch={})='
                            'ER:{:.3f} F:{:.3f} LR:{:.3f} AngE:{:.2f} AngAcc:{:.3f} Dist:{:.2f} RelDist:{:.2f} SELD:{:.2f}'
                        ).format(
                            epoch_cnt, val_mode, train_time, val_time,
                            train_loss, val_loss,
                            val_ER, val_F, val_LR,
                            val_LE, val_ang_acc,
                            val_dist_err, val_rel_dist_err, val_seld_scr,
                            best_val_epoch, best_val_mode,
                            best_ER, best_F, best_LR, best_LE, best_ang_acc, best_dist_err, best_rel_dist_err, best_seld_scr,
                            best_full_epoch,
                            best_full_ER, best_full_F, best_full_LR, best_full_LE, best_full_ang_acc, best_full_dist_err, best_full_rel_dist_err, best_full_seld_scr
                        )
                        epoch_iterator.set_postfix(train_loss='{:.4f}'.format(train_loss), val_loss='{:.4f}'.format(val_loss), seld='{:.2f}'.format(val_seld_scr))
                        tqdm.write(epoch_summary)

                        should_stop = run_full_val and patience_cnt > params['patience']
                        if should_stop:
                            tqdm.write('Early stopping triggered at epoch {}'.format(epoch_cnt))

                        write_sync_signal(
                            result_signal,
                            {
                                'should_stop': bool(should_stop),
                                'val_mode': val_mode,
                                'epoch': int(epoch_cnt),
                            }
                        )
                    else:
                        should_stop = bool(wait_for_sync_signal(result_signal).get('should_stop', False))

                    if should_stop:
                        break

                # ---------------------------------------------------------------------
                # Evaluate on unseen test data
                # ---------------------------------------------------------------------
                dist_tqdm_write(dist_ctx, 'Loading best-any validation model weights from {}'.format(model_name))
                load_model_state(model, model_name, strict=False)

                dist_tqdm_write(dist_ctx, 'Loading unseen test dataset')
                run_ddp_eval = dist_ctx['enabled'] and bool(params.get('ddp_eval', False))
                data_gen_test, test_filelist_full, test_filelist_local = create_data_generator(
                    params=params, split=test_splits[split_cnt], dist_ctx=dist_ctx, shuffle=False, per_file=True, shard=run_ddp_eval
                )

                # Dump results in DCASE output format for calculating final scores
                dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
                if is_main_process(dist_ctx):
                    cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
                    tqdm.write(
                        'Test predictions -> {} (global/local = {}/{})'.format(
                            dcase_output_test_folder, len(test_filelist_full), len(test_filelist_local)
                        )
                    )
                test_folder_ready_signal = sync_signal_path(sync_dir, 'test_folder_ready')
                test_result_signal = sync_signal_path(sync_dir, 'test_result')
                if is_main_process(dist_ctx):
                    remove_sync_signal(test_folder_ready_signal)
                    remove_sync_signal(test_result_signal)
                    write_sync_signal(test_folder_ready_signal, {'ready': True})
                elif run_ddp_eval:
                    wait_for_sync_signal(test_folder_ready_signal)

                if run_ddp_eval:
                    test_loss = test_epoch(
                        data_gen_test, model, criterion, dcase_output_test_folder, params, device, dist_ctx=dist_ctx, desc='Test'
                    )
                else:
                    test_loss = None
                    if is_main_process(dist_ctx):
                        full_test_gen = cls_data_generator.DataGenerator(
                            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
                        )
                        test_loss = test_epoch(
                            full_test_gen, unwrap_model(model), criterion, dcase_output_test_folder, params, device, dist_ctx=None, desc='Test'
                        )

                if is_main_process(dist_ctx):
                    tqdm.write('Scoring test outputs from {}'.format(dcase_output_test_folder))
                    use_jackknife = True
                    test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(
                        dcase_output_test_folder, is_jackknife=use_jackknife
                    )

                    test_ang = test_LE[0] if use_jackknife else test_LE
                    test_ang_acc = max(0.0, 1.0 - (test_ang / 180.0)) if np.isfinite(test_ang) else np.nan
                    tqdm.write('Test loss: {:.4f}'.format(test_loss))
                    tqdm.write('SELD score: {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
                    tqdm.write('SED ER/F-score: {:0.3f} / {:0.1f} {}'.format(test_ER[0] if use_jackknife else test_ER, 100 * test_F[0] if use_jackknife else 100 * test_F, '[{:0.2f}, {:0.2f}]'.format(100 * test_F[1][0], 100 * test_F[1][1]) if use_jackknife else ''))
                    tqdm.write('Localization recall: {:0.3f} {}'.format(test_LR[0] if use_jackknife else test_LR, '[{:0.2f}, {:0.2f}]'.format(test_LR[1][0], test_LR[1][1]) if use_jackknife else ''))
                    tqdm.write('AngE_deg / AngAcc: {:0.1f} / {:0.3f} {}'.format(test_ang, test_ang_acc, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else ''))
                    tqdm.write('Distance error: {:0.2f} {}'.format(test_dist_err[0] if use_jackknife else test_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_dist_err[1][0], test_dist_err[1][1]) if use_jackknife else ''))
                    tqdm.write('Relative distance error: {:0.2f} {}'.format(test_rel_dist_err[0] if use_jackknife else test_rel_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_rel_dist_err[1][0], test_rel_dist_err[1][1]) if use_jackknife else ''))

                    if params['average'] == 'macro':
                        tqdm.write('Classwise results on unseen test data')
                        tqdm.write('Class\tF\tAngE_deg\tdist_err\treldist_err\tSELD_score')
                        for cls_cnt in range(params['unique_classes']):
                            tqdm.write('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                                cls_cnt,

                                classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                            classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                                classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                            classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                                classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                            classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                                classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                            classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else '',

                                classwise_test_scr[0][6][cls_cnt] if use_jackknife else classwise_test_scr[6][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][6][cls_cnt][0],
                                                            classwise_test_scr[1][6][cls_cnt][1]) if use_jackknife else ''))
                    write_sync_signal(test_result_signal, {'done': True})
                else:
                    wait_for_sync_signal(test_result_signal)

        if params['mode'] == 'eval':

            dist_tqdm_write(dist_ctx, 'Loading evaluation dataset')
            run_ddp_eval = dist_ctx['enabled'] and bool(params.get('ddp_eval', False))
            data_gen_eval, eval_filelist_full, eval_filelist_local = create_data_generator(
                params=params, split=1, dist_ctx=dist_ctx, shuffle=False, per_file=True, is_eval=True, shard=run_ddp_eval
            )

            if params['modality'] == 'audio_visual':
                data_in, vid_data_in, data_out = data_gen_eval.get_data_sizes()
                base_model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in)
            else:
                data_in, data_out = data_gen_eval.get_data_sizes()
                base_model = seldnet_model.SeldModel(data_in, data_out, params)

            model = maybe_wrap_model_for_multi_gpu(base_model, params, dist_ctx)

            dist_tqdm_write(dist_ctx, 'Loading best model weights')
            model_name = os.path.join(params['model_dir'], '3_1_dev_split0_multiaccdoa_foa_model.h5')
            load_model_state(model, model_name, strict=False)

            # Dump results in DCASE output format for calculating final scores
            loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

            dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_{}_eval'.format(params['dataset'], loc_output, strftime("%Y%m%d%H%M%S", gmtime())))
            sync_dir = os.path.join(params['dcase_output_dir'], '{}_{}_eval_sync'.format(params['dataset'], loc_output))
            if is_main_process(dist_ctx):
                cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
                cls_feature_class.delete_and_create_folder(sync_dir)
                tqdm.write(
                    'Eval predictions -> {} (global/local = {}/{})'.format(
                        dcase_output_test_folder, len(eval_filelist_full), len(eval_filelist_local)
                    )
                )
            else:
                init_run_sync_dir(sync_dir)
            eval_folder_ready_signal = sync_signal_path(sync_dir, 'eval_folder_ready')
            eval_result_signal = sync_signal_path(sync_dir, 'eval_result')
            if is_main_process(dist_ctx):
                remove_sync_signal(eval_folder_ready_signal)
                remove_sync_signal(eval_result_signal)
                write_sync_signal(eval_folder_ready_signal, {'ready': True})
            elif run_ddp_eval:
                wait_for_sync_signal(eval_folder_ready_signal)

            if run_ddp_eval:
                eval_epoch(data_gen_eval, model, dcase_output_test_folder, params, device, dist_ctx=dist_ctx, desc='Eval')
            else:
                if is_main_process(dist_ctx):
                    full_eval_gen = cls_data_generator.DataGenerator(
                        params=params, shuffle=False, per_file=True, is_eval=True
                    )
                    eval_epoch(full_eval_gen, unwrap_model(model), dcase_output_test_folder, params, device, dist_ctx=None, desc='Eval')
                    write_sync_signal(eval_result_signal, {'done': True})
                else:
                    wait_for_sync_signal(eval_result_signal)
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
