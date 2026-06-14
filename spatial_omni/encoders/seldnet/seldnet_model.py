# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed


class MSELoss_ADPIT(object):
    def __init__(self, params=None):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')
        params = params or {}
        self._pos_weight = float(params.get('adpit_pos_weight', 1.0))
        self._neg_weight = float(params.get('adpit_neg_weight', 1.0))
        self._dynamic_weight = bool(params.get('adpit_dynamic_weight', False))
        self._dynamic_pos_cap = float(params.get('adpit_dynamic_pos_cap', 1.0))
        self._activity_aux_weight = float(params.get('activity_aux_weight', 0.0))
        self._activity_aux_pos_margin = float(params.get('activity_aux_pos_margin', 0.5))
        self._activity_aux_neg_margin = float(params.get('activity_aux_neg_margin', 0.1))
        self._distance_loss_weight = float(params.get('distance_loss_weight', 1.0))
        self._eps = 1e-8
        self._regression_scale = 1.0
        self._distance_scale = 1.0

    def set_scales(self, regression_scale=1.0, distance_scale=1.0):
        self._regression_scale = float(regression_scale)
        self._distance_scale = float(distance_scale)

    def _compute_positive_scale(self, active_mask, loss_mask):
        if not self._dynamic_weight:
            return active_mask.new_tensor(self._pos_weight)
        pos_count = (active_mask * loss_mask).sum()
        neg_count = ((1.0 - active_mask) * loss_mask).sum()
        balance = torch.sqrt((neg_count + self._eps) / (pos_count + self._eps))
        balance = torch.clamp(balance, min=1.0, max=self._dynamic_pos_cap)
        return active_mask.new_tensor(self._pos_weight) * balance

    def _each_calc(self, output, target, activity_target):
        output_tracks = output.reshape(output.shape[0], output.shape[1], 3, 4, output.shape[-1])
        target_tracks = target.reshape(target.shape[0], target.shape[1], 3, 4, target.shape[-1])
        activity_target = activity_target.reshape(activity_target.shape[0], activity_target.shape[1], 3, activity_target.shape[-1]).to(target.dtype)
        inactive_target = 1.0 - activity_target

        xyz_out = output_tracks[:, :, :, :3, :]
        xyz_tgt = target_tracks[:, :, :, :3, :]
        dist_out = output_tracks[:, :, :, 3, :]
        dist_tgt = target_tracks[:, :, :, 3, :]
        norms = torch.sqrt(torch.clamp((xyz_out ** 2).sum(dim=3), min=self._eps))

        xyz_sqerr = ((xyz_out - xyz_tgt) ** 2).sum(dim=3)
        active_xyz_count = activity_target.sum(dim=2).clamp_min(1.0)
        active_xyz_loss = (xyz_sqerr * activity_target).sum(dim=2) / active_xyz_count

        valid_dist = ((dist_tgt >= 0).to(target.dtype) * activity_target)
        valid_dist_count = valid_dist.sum(dim=2).clamp_min(1.0)
        dist_sqerr = (dist_out - dist_tgt) ** 2
        active_dist_loss = (dist_sqerr * valid_dist).sum(dim=2) / valid_dist_count

        pos_scale = self._compute_positive_scale(activity_target, torch.ones_like(activity_target))
        loss = pos_scale * active_xyz_loss + pos_scale * self._distance_loss_weight * self._distance_scale * active_dist_loss

        if self._activity_aux_weight > 0.0:
            active_margin_loss = (torch.relu(self._activity_aux_pos_margin - norms) ** 2 * activity_target).sum(dim=2) / active_xyz_count
            loss = loss + self._activity_aux_weight * pos_scale * active_margin_loss

        inactive_count = inactive_target.sum(dim=2).clamp_min(1.0)
        inactive_reg = (torch.relu(norms - self._activity_aux_neg_margin) ** 2 * inactive_target).sum(dim=2) / inactive_count
        loss = loss + self._neg_weight * inactive_reg
        return self._regression_scale * loss

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*4*13]
            target: [batch_size, frames, num_track_dummy=6, num_axis=5, num_class=13]
        Return:
            loss: scalar
        """
        act_A0 = target[:, :, 0, 0:1, :]
        act_B0 = target[:, :, 1, 0:1, :]
        act_B1 = target[:, :, 2, 0:1, :]
        act_C0 = target[:, :, 3, 0:1, :]
        act_C1 = target[:, :, 4, 0:1, :]
        act_C2 = target[:, :, 5, 0:1, :]

        target_A0 = act_A0 * target[:, :, 0, 1:, :]
        target_B0 = act_B0 * target[:, :, 1, 1:, :]
        target_B1 = act_B1 * target[:, :, 2, 1:, :]
        target_C0 = act_C0 * target[:, :, 3, 1:, :]
        target_C1 = act_C1 * target[:, :, 4, 1:, :]
        target_C2 = act_C2 * target[:, :, 5, 1:, :]

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        act_A0A0A0 = torch.cat((act_A0, act_A0, act_A0), 2)
        act_B0B0B1 = torch.cat((act_B0, act_B0, act_B1), 2)
        act_B0B1B0 = torch.cat((act_B0, act_B1, act_B0), 2)
        act_B0B1B1 = torch.cat((act_B0, act_B1, act_B1), 2)
        act_B1B0B0 = torch.cat((act_B1, act_B0, act_B0), 2)
        act_B1B0B1 = torch.cat((act_B1, act_B0, act_B1), 2)
        act_B1B1B0 = torch.cat((act_B1, act_B1, act_B0), 2)
        act_C0C1C2 = torch.cat((act_C0, act_C1, act_C2), 2)
        act_C0C2C1 = torch.cat((act_C0, act_C2, act_C1), 2)
        act_C1C0C2 = torch.cat((act_C1, act_C0, act_C2), 2)
        act_C1C2C0 = torch.cat((act_C1, act_C2, act_C0), 2)
        act_C2C0C1 = torch.cat((act_C2, act_C0, act_C1), 2)
        act_C2C1C0 = torch.cat((act_C2, act_C1, act_C0), 2)

        valid_A = act_A0.squeeze(2) > 0
        valid_B = (act_B0 + act_B1).squeeze(2) > 0
        valid_C = (act_C0 + act_C1 + act_C2).squeeze(2) > 0

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])
        loss_0 = self._each_calc(output, target_A0A0A0, act_A0A0A0)
        loss_1 = self._each_calc(output, target_B0B0B1, act_B0B0B1)
        loss_2 = self._each_calc(output, target_B0B1B0, act_B0B1B0)
        loss_3 = self._each_calc(output, target_B0B1B1, act_B0B1B1)
        loss_4 = self._each_calc(output, target_B1B0B0, act_B1B0B0)
        loss_5 = self._each_calc(output, target_B1B0B1, act_B1B0B1)
        loss_6 = self._each_calc(output, target_B1B1B0, act_B1B1B0)
        loss_7 = self._each_calc(output, target_C0C1C2, act_C0C1C2)
        loss_8 = self._each_calc(output, target_C0C2C1, act_C0C2C1)
        loss_9 = self._each_calc(output, target_C1C0C2, act_C1C0C2)
        loss_10 = self._each_calc(output, target_C1C2C0, act_C1C2C0)
        loss_11 = self._each_calc(output, target_C2C0C1, act_C2C0C1)
        loss_12 = self._each_calc(output, target_C2C1C0, act_C2C1C0)

        invalid_penalty = output.new_tensor(1e6)
        loss_stack = torch.stack((
            loss_0 + (~valid_A).to(loss_0.dtype) * invalid_penalty,
            loss_1 + (~valid_B).to(loss_1.dtype) * invalid_penalty,
            loss_2 + (~valid_B).to(loss_2.dtype) * invalid_penalty,
            loss_3 + (~valid_B).to(loss_3.dtype) * invalid_penalty,
            loss_4 + (~valid_B).to(loss_4.dtype) * invalid_penalty,
            loss_5 + (~valid_B).to(loss_5.dtype) * invalid_penalty,
            loss_6 + (~valid_B).to(loss_6.dtype) * invalid_penalty,
            loss_7 + (~valid_C).to(loss_7.dtype) * invalid_penalty,
            loss_8 + (~valid_C).to(loss_8.dtype) * invalid_penalty,
            loss_9 + (~valid_C).to(loss_9.dtype) * invalid_penalty,
            loss_10 + (~valid_C).to(loss_10.dtype) * invalid_penalty,
            loss_11 + (~valid_C).to(loss_11.dtype) * invalid_penalty,
            loss_12 + (~valid_C).to(loss_12.dtype) * invalid_penalty,
        ), dim=0)
        loss_min = torch.min(loss_stack, dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss


class SEDDOAMultiTaskLoss(object):
    def __init__(self, params=None):
        super().__init__()
        params = params or {}
        self._regression_loss = MSELoss_ADPIT(params)
        self._sed_pos_weight = float(params.get('sed_pos_weight', 1.0))
        self._sed_neg_weight = float(params.get('sed_neg_weight', 1.0))
        self._sed_dynamic_weight = bool(params.get('sed_dynamic_weight', False))
        self._sed_pos_cap = float(params.get('sed_dynamic_pos_cap', 20.0))
        self._sed_warmup_epochs = int(params.get('sed_warmup_epochs', 0))
        self._sed_warmup_weight = float(params.get('sed_warmup_weight', 1.0))
        self._sed_main_weight = float(params.get('sed_main_weight', 1.0))
        self._reg_warmup_weight = float(params.get('regression_warmup_weight', 1.0))
        self._reg_main_weight = float(params.get('regression_main_weight', 1.0))
        self._dist_warmup_scale = float(params.get('distance_warmup_scale', 1.0))
        self._dist_main_scale = float(params.get('distance_main_scale', 1.0))
        self._epoch = 0
        self._eps = 1e-8

    def set_epoch(self, epoch):
        self._epoch = int(epoch)
        warmup = self._epoch < self._sed_warmup_epochs
        reg_scale = self._reg_warmup_weight if warmup else self._reg_main_weight
        dist_scale = self._dist_warmup_scale if warmup else self._dist_main_scale
        self._regression_loss.set_scales(reg_scale, dist_scale)

    def _compute_sed_weights(self, sed_target):
        if not self._sed_dynamic_weight:
            return sed_target.new_full(sed_target.shape, self._sed_pos_weight)
        pos_count = sed_target.sum()
        neg_count = (1.0 - sed_target).sum()
        balance = torch.sqrt((neg_count + self._eps) / (pos_count + self._eps))
        balance = torch.clamp(balance, min=1.0, max=self._sed_pos_cap)
        return sed_target.new_full(sed_target.shape, self._sed_pos_weight * balance)

    def __call__(self, output, target):
        accdoa_output = output['accdoa']
        sed_logits = output['sed_logits']

        sed_target = target[:, :, :, 0, :].amax(dim=2).to(accdoa_output.dtype)
        pos_weights = self._compute_sed_weights(sed_target)
        sed_weights = sed_target * pos_weights + (1.0 - sed_target) * self._sed_neg_weight
        sed_bce = F.binary_cross_entropy_with_logits(sed_logits, sed_target, reduction='none')
        sed_loss = (sed_weights * sed_bce).mean()

        reg_loss = self._regression_loss(accdoa_output, target)
        sed_weight = self._sed_warmup_weight if self._epoch < self._sed_warmup_epochs else self._sed_main_weight
        return sed_weight * sed_loss + reg_loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params=params
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        # fusion layers
        if in_vid_feat_shape is not None:
            self.visual_embed_to_d_model = nn.Linear(in_features = int(in_vid_feat_shape[2]*in_vid_feat_shape[3]), out_features = self.params['rnn_size'] )
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.params['rnn_size'], nhead=self.params['nb_heads'], batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.params['nb_transformer_layers'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))
        self.use_explicit_sed_head = bool(params.get('explicit_sed_head', False))
        if self.use_explicit_sed_head:
            self.sed_head = nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], self.nb_classes, bias=True)

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x)
        if self.use_explicit_sed_head:
            return {'accdoa': doa, 'sed_logits': self.sed_head(x)}

        # the below-commented code applies tanh for doa and relu for distance estimates respectively in multi-accdoa scenarios.
        # they can be uncommented and used, but there is no significant changes in the results.
        #doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, 13)
        #doa1 = doa[:, :, :, :3, :]
        #dist = doa[:, :, :, 3:, :]

        #doa1 = self.doa_act(doa1)
        #dist = self.dist_act(dist)
        #doa2 = torch.cat((doa1, dist), dim=3)

        #doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        #return doa2
        return doa
