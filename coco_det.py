import os

crop_size = (
    800,
    800
)

custom_imports = dict(
    imports=['mmpretrain.models'],
    allow_failed_imports=False)

dataset_type = 'CocoDataset'  # 目标检测数据类型

data_root = r'F:/PythonProject/Skysense++/SkySensePlusPlus-main/rs_datasets/coco_det'  # 数据存放路径

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # 归一化设置参数

# 工作目录
work_dir = 'save/coco_dect'

# 训练的数据处理流程
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),  # 第一步，加载图像
    dict(type='LoadAnnotations', with_bbox=True, poly2mask=False),  # 第二步，对当前图像，加载标注信息
    dict(type='Resize', scale_factor=(1.0, 1.0), keep_ratio=True), # 第三步，对图像进行resize
    dict(type='RandomFlip', prob=0.5),  # 第四步，随机翻转
    dict(type='Pad', size_divisor=32),  # 第六步，边缘扩充，变为32的倍数
    dict(type='PackDetInputs'),
]

# 测试的数据处理流程
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=(1.0, 1.0),
        transforms=[
            dict(type='Resize', scale_factor=(1.0, 1.0), keep_ratio=True),
            dict(type='RandomFlip', prob=0.1),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 需要进行目标检测的类型
classes = ("ship")

# 训练集数据加载参数
train_dataloader = dict(  # 训练 dataloader 配置
    batch_size=2,  # 单个 GPU 的 batch size
    num_workers=1,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict(  # 训练数据的采样器
        type='DefaultSampler',  # 默认的采样器，同时支持分布式和非分布式训练。
        shuffle=True),  # 随机打乱每个轮次训练数据的顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,
        data_root=data_root,
        ann_file=os.path.join(data_root, 'train.json'),  # 标注文件路径
        data_prefix=dict(img=os.path.join(data_root,'train', '')),  # 图片路径前缀
        filter_cfg=dict(filter_empty_gt=False, min_size=0),  # 图片和标注的过滤配置，过滤掉没有标注的空样本，以及，目标小于5个像素的物体
        pipeline=train_pipeline))  # 这是由之前创建的 train_pipeline 定义的数据处理流程。

# 验证集数据加载参数     
val_dataloader = dict(  # 验证 dataloader 配置
    batch_size=1,  # 单个 GPU 的 Batch size。如果 batch-szie > 1，组成 batch 时的额外填充会影响模型推理精度
    num_workers=1,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False,  # 是否丢弃最后未能组成一个批次的数据
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=r"F:\PythonProject\Skysense++\SkySensePlusPlus-main\rs_datasets\coco_det\\train1.json",  # 标注文件路径
        data_prefix=dict(img=os.path.join(data_root, 'val', '')),
        test_mode=True,  # 开启测试模式，避免数据集过滤图片和标注
        pipeline=test_pipeline))
        
# 测试集数据加载参数
test_dataloader=val_dataloader

# 验集的测评器参数
val_evaluator = dict(  # 验证过程使用的评测器
    type='CocoMetric',  # 用于评估检测和实例分割的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=os.path.join(data_root, 'train1.json'),  # 标注文件路径
    metric=['bbox', 'segm'],  # 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    format_only=False)

# 测试的测评器参数xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx待完善
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric=['bbox', 'segm'],
    format_only=True,  # 只将模型输出转换为 coco 的 JSON 格式并保存
    outfile_prefix='./work_dirs/coco_detection/test')  # 要保存的 JSON 文件的前缀
    
# 训练循环的类型
train_cfg = dict(max_iters=50000, type='IterBasedTrainLoop', val_interval=500)
    
# 验证的设置
val_cfg = dict(type='ValLoop')

# 测试的设置
test_cfg = dict(type='TestLoop')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# 优化器封装的配置
optim_wrapper = dict(  
    type='OptimWrapper',  # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=optimizer,
    clip_grad=None,  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪
    )

# 推理结果的可视化器
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

# 分布式训练的启动器
launcher = None  # pytorch

# 默认注册域，优先从这里加载文件
default_scope = 'mmdet'

# 训练日志
log_level = 'INFO'
log_processor = dict(by_epoch=False)

# 加载预训练权重
load_from = None

# 默认钩子配置
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, 
        interval=1000, 
        save_best='mAP',        # 改为检测核心指标 mAP
        max_keep_ckpts=1,
        type='CheckpointHook'),
    logger=dict(
        interval=50, 
        log_metric_by_epoch=False, 
        type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook')  # 改为检测可视化钩子
)

# 运行相关的配置
env_cfg = dict(
    cudnn_benchmark=False,  # 是否启用 cudnn benchmark
    mp_cfg=dict(  # 多进程设置
        mp_start_method='fork',  # 使用 fork 来启动多进程。
        opencv_num_threads=0),  # 关闭 opencv 的多线程以避免系统超负荷
    dist_cfg=dict(backend='nccl'),  # 分布式相关设置
)

# 是否从 `load_from` 中定义的检查点恢复，做微调用的
resume = False

# 固定随机种子
randomness = dict(seed=20240315)

# 验证、测试时tta
tta_model = dict(type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),   # 测试/验证时加载 GT（检测任务会读取 gt_bboxes 和 gt_labels）
            ],
            [
                dict(type='PackDetInputs'),     # 打包成检测模型所需的输入格式
            ],
        ],
        type='TestTimeAug'),
]

# 将参数调度器修改为iter-based
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

# 模型
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
norm_cfg = dict(requires_grad=True, type='SyncBN')
custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)
# 模型配置
# 模型
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    # 骨干网络：使用ViT-small（修正后的正确配置）
    backbone=dict(
        arch='base',
        img_size=crop_size,
        in_channels=3,
        interpolate_mode='bilinear',
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        out_indices=(
            1,#5,
            3,#11,
            5,#17,
            7,#23,
        ),
        patch_size=4,
        qkv_bias=True,
        type='mmpretrain.VisionTransformer',
        ),
    
    # 颈部网络：FPN
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],  # ViT-small的隐藏维度为384
        out_channels=256,  # FPN标准输出通道数
        num_outs=5,  # 输出5个尺度的特征图
        start_level=0,  # 从第0层开始
        add_extra_convs='on_input',  # 在输入上添加额外卷积层
        relu_before_extra_convs=True),  # 在额外卷积前使用ReLU
    
    # RPN头
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,  # FPN输出通道数
        feat_channels=256,  # 特征通道数
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],  # 基础尺度
            ratios=[0.5, 1.0, 2.0],  # 宽高比
            strides=[4, 8, 16, 32, 64]),  # 每个特征图的步长
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    
    # ROI头
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # 您的类别数，"ship"一个类别
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    
    # 训练和测试配置
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
        max_iters=50000, 
        type='IterBasedTrainLoop', 
        val_interval=500),
    
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100),
        type='TestLoop'
    )
    val_cfg=dict(type='ValLoop')
)