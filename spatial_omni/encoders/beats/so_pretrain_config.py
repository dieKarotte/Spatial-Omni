"""Default stage-1 training presets for SO-Encoder on ov1/ov2/ov3 FOA data.

This file keeps the actual training config separate from the generic training
utilities in ``train_so_pretrain.py`` so the user can inspect and edit a
single concrete preset before launching training.
"""

from .train_so_pretrain import (
    DEFAULT_OV1_MANIFEST,
    DEFAULT_OV2_MANIFEST,
    DEFAULT_OV3_MANIFEST,
    TrainSOBackboneConfig,
    make_ov1_ast_balanced_config,
    make_ov1_ast_classwarmup_config,
    make_ov1_ast_config,
    make_ov1_ast_spatial_config,
    make_ov1_pretrunk_ast_class_config,
    make_ov1_pretrunk_ast_phase0_config,
    make_ov1_pretrunk_ast_spatial_config,
    make_ov1_spatial_finetune_config,
    make_ov1_stage1_config,
    make_ov123_stage1_config,
    make_ov123_spatial_finetune_config,
    make_ov23_stage1_config,
    make_ov23_spatial_finetune_config,
)


OV1_STAGE1_CFG: TrainSOBackboneConfig = make_ov1_stage1_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV123_STAGE1_CFG: TrainSOBackboneConfig = make_ov123_stage1_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
    ov2_manifest_path=DEFAULT_OV2_MANIFEST,
    ov3_manifest_path=DEFAULT_OV3_MANIFEST,
)


OV23_STAGE1_CFG: TrainSOBackboneConfig = make_ov23_stage1_config(
    ov2_manifest_path=DEFAULT_OV2_MANIFEST,
    ov3_manifest_path=DEFAULT_OV3_MANIFEST,
)


OV1_SPATIAL_FINETUNE_CFG: TrainSOBackboneConfig = make_ov1_spatial_finetune_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV1_AST_CFG: TrainSOBackboneConfig = make_ov1_ast_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV1_AST_CLASSWARMUP_CFG: TrainSOBackboneConfig = make_ov1_ast_classwarmup_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV1_AST_SPATIAL_CFG: TrainSOBackboneConfig = make_ov1_ast_spatial_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV1_AST_BALANCED_CFG: TrainSOBackboneConfig = make_ov1_ast_balanced_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV1_PRETRUNK_AST_CLASS_CFG: TrainSOBackboneConfig = make_ov1_pretrunk_ast_class_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV1_PRETRUNK_AST_PHASE0_CFG: TrainSOBackboneConfig = make_ov1_pretrunk_ast_phase0_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV1_PRETRUNK_AST_SPATIAL_CFG: TrainSOBackboneConfig = make_ov1_pretrunk_ast_spatial_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
)


OV123_SPATIAL_FINETUNE_CFG: TrainSOBackboneConfig = make_ov123_spatial_finetune_config(
    ov1_manifest_path=DEFAULT_OV1_MANIFEST,
    ov2_manifest_path=DEFAULT_OV2_MANIFEST,
    ov3_manifest_path=DEFAULT_OV3_MANIFEST,
)


OV23_SPATIAL_FINETUNE_CFG: TrainSOBackboneConfig = make_ov23_spatial_finetune_config(
    ov2_manifest_path=DEFAULT_OV2_MANIFEST,
    ov3_manifest_path=DEFAULT_OV3_MANIFEST,
)
