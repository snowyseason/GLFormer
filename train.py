import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.conformer_cross_plus import mctconformer_small_patch16_crossplus, mctconformer_tiny_patch16_crossplus, mctconformer_base_patch16_crossplus
from utils.utils import (download_weights, get_lr_scheduler,
                         set_optimizer_lr, show_config, weights_init,
                         set_random_seed)
from utils.callbacks import LossHistory
from utils.dataloader import get_dataset
from utils_fit import fit_one_epoch, eval_testset
import argparse
# ------------------------------------------------------#
#   获取所需的超参数
# ------------------------------------------------------#
parser = argparse.ArgumentParser()

# Common hyperparameters
parser.add_argument('--Cuda', type=bool, default='True',
                    help='是否使用Cuda,没有GPU可以设置成False')
parser.add_argument('--Init_Epoch', default=0,
                    help='模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100')
parser.add_argument('--Freeze_Train', default=False,
                    help='是否进行冻结训练,默认先冻结主干训练后解冻训练。')
parser.add_argument('--Freeze_Epoch', default=10,type= int,
                    help='模型冻结训练的Freeze_Epoch(当Freeze_Train=False时失效)')
parser.add_argument('--Freeze_batch_size', default=32,type= int,
                    help='模型冻结训练的batch_size')
parser.add_argument('--UnFreeze_Epoch', type = int, default=50, help='模型总共训练的epoch')
parser.add_argument('--Unfreeze_batch_size', type= int, default=32, help='模型在解冻后的batch_size')
parser.add_argument('--save_dir', default='logs/c45v2_hornet_tiny7x7_shapley234mul_cpk01', help='模型保存的路径')
parser.add_argument('--seed', default=666, help='随机数种子')
parser.add_argument('--num_workers', default=4, help='多线程加载')
parser.add_argument('--save_period', default=None, help='保存轮数间隔')

# model
parser.add_argument('--wsol_method', default='GLformer', help='weakly methods')
parser.add_argument('--backbone', default='hornet_tiny_7x7', help='所用模型种类')
parser.add_argument('--pretrained', default=True, help='是否加载预训练权重')
parser.add_argument('--model_path', default="", help='模型加载权重的路径')
parser.add_argument('--num_classes', type=int, default=16, help='分类的类别数')
parser.add_argument('--input_shape', default=(256, 256), help='图像的高宽')

# data
parser.add_argument('--dataset_name', type=str, default='C45V2')
parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                    default='datasets/',
                    help='path to dataset images')
parser.add_argument('--metadata_root', type=str, default='metadata/')
parser.add_argument('--mask_root', metavar='/PATH/TO/MASKS',
                    default='dataset/',
                    help='path to masks')
parser.add_argument('--proxy_training_set', nargs='?',
                    const=True, default=False,
                    help='Efficient hyper_parameter search with a proxy '
                            'training set.(用一个辅助的训练集进行高效的超参数搜索)')
parser.add_argument('--num_val_sample_per_class', type=int, default=0,
                    help='Number of full_supervision validation sample per '
                            'class. 0 means "use all available samples".')

# setting
parser.add_argument('--multi_contour_eval', nargs='?',
                    const=True, default=True)
parser.add_argument('--multi_iou_eval', nargs='?',
                    const=True, default=True)
parser.add_argument('--cam_curve_interval', type=float, default=.001,
                    help='CAM curve interval')
parser.add_argument('--resize_size', type=int, default=256,
                    help='input resize size')
parser.add_argument('--crop_size', type=int, default=224,
                    help='input crop size')
parser.add_argument('--iou_threshold_list', nargs='+',
                    type=int, default=[30, 50, 70])
parser.add_argument('--eval_checkpoint_type', type=str, default='best',
                    choices=('best', 'last'))
parser.add_argument('--box_v2_metric', nargs='?',
                    const=True, default=True)
parser.add_argument('--first_eval', nargs='?',
                    const=True, default=False)
parser.add_argument('--eval_period', type=int, default=50)

parser.add_argument('--cam_batch_size', type=int, default=1, help='生成shapley类激活图的batch size')
parser.add_argument('--interval', default=1, help='mask间隔')
parser.add_argument('--n_samples', default=10, help='shapley values估计采样的次数')
parser.add_argument('--perturbations_per_eval', default=10000, help='一次扰动的次数')

# optimizer
parser.add_argument('--optimizer_type', default='sgd', help='优化器种类')
parser.add_argument('--momentum', default=0.9, help='动量')
parser.add_argument('--weight_decay', default=5e-4, help='权重衰减')
parser.add_argument('--lr_decay_type', default='step', help='学习率下降')
parser.add_argument('--Init_lr',type=float,default=3e-5, help='初始学习率')
parser.add_argument('--Min_lr',  type=float,default=5e-6, help='最小学习率')
parser.add_argument('--warmup_mode', default='linear', help='预热策略')

parser.add_argument("--num_cct", default=2, type=int)
parser.add_argument('--patch_attn_refine', type=bool, default=False)
parser.add_argument('--conv_attn_refine', type=bool, default=False)
parser.add_argument('--attention_type', type=str, default='mct')
parser.add_argument('--remark', default=None)

args = parser.parse_args()
# ------------------------------------------------------#
#   设置用到的显卡
# ------------------------------------------------------#
ngpus_per_node = torch.cuda.device_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained = args.pretrained
backbone = args.backbone

# ------------------------------------------------------#
#   pretrained 下载预训练权重
# ------------------------------------------------------#
if pretrained:
    download_weights(backbone)

num_classes = args.num_classes
input_shape = args.input_shape
crop_shape = [args.crop_size, args.crop_size]

model_dict = {
    'conformer-t': mctconformer_tiny_patch16_crossplus,
    'conformer-s': mctconformer_small_patch16_crossplus,
    'conformer-b': mctconformer_base_patch16_crossplus
}
model = model_dict[args.backbone](num_classes=args.num_classes,pretrained=True,num_cross_layers = args.num_cct)

# ------------------------------------------------------#
#   模型初始化
# ------------------------------------------------------#
if not pretrained:
    weights_init(model)

model_path = args.model_path
if model_path != "":
    print('Load weights {}.'.format(model_path))
    # ------------------------------------------------------#
    #   根据预训练权重的Key和模型的Key进行加载
    # ------------------------------------------------------#
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    # ------------------------------------------------------#
    #   显示没有匹配上的Key
    # ------------------------------------------------------#
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
# ----------------------#
#   初始化loss_history用于记录loss
# ----------------------#
loss_history = LossHistory(args.save_dir, model, input_shape=input_shape)

# ----------------------#
#   模型设置为训练模式
# ----------------------#
model_train = model.train()
if args.Cuda:
    # model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()

# ----------------------#
#   设置随机数种子
# ----------------------#
set_random_seed(args.seed)
datasets = get_dataset(
    data_roots=args.data_root,
    metadata_root=args.metadata_root,
    resize_size=args.resize_size,
    crop_size=args.crop_size,
    proxy_training_set=args.proxy_training_set,
    num_val_sample_per_class=args.num_val_sample_per_class,
    dataset_name=args.dataset_name
)
train_dataset = datasets['train']
val_dataset = datasets['val']
test_dataset = datasets['test']
num_train = len(train_dataset)
num_val = len(val_dataset)
num_test = len(test_dataset)

args_dict = vars(args)
args_dict['num_train'] = num_train
args_dict['num_val'] = num_val
args_dict['num_test'] = num_test
show_config(**args_dict)

# ------------------------------------------------------#
#   主干特征提取网络特征通用，冻结训练可以加快训练速度
#   也可以在训练初期防止权值被破坏。
#   Init_Epoch为起始世代
#   Freeze_Epoch为冻结训练的世代
#   UnFreeze_Epoch总训练世代
#   提示OOM或者显存不足请调小Batch_size
# ------------------------------------------------------#
if True:

    batch_size = args.Freeze_batch_size if args.Freeze_Train else args.Unfreeze_batch_size
    num_workers = args.num_workers

    Init_lr_fit = args.Init_lr
    Min_lr_fit = args.Min_lr
    optimizer = {
        'adam': optim.Adam(model_train.parameters(), Init_lr_fit, betas=(args.momentum, 0.999),
                           weight_decay=args.weight_decay),
        'sgd': optim.SGD(model_train.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True)
    }[args.optimizer_type]

    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.UnFreeze_Epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    epoch_step_cam = num_val // args.cam_batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    train_sampler = None
    val_sampler = None
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=False, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=False, sampler=val_sampler)

    cam_gen_val = DataLoader(val_dataset, shuffle=False, batch_size=args.cam_batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False, sampler=val_sampler)

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    for epoch in range(args.Init_Epoch, args.UnFreeze_Epoch):

        Init_lr_fit = args.Init_lr
        Min_lr_fit = args.Min_lr

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.UnFreeze_Epoch)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, drop_last=False, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, drop_last=False, sampler=val_sampler)
        cam_gen_val = DataLoader(val_dataset, shuffle=False, batch_size=args.cam_batch_size,
                                    num_workers=num_workers, pin_memory=True,
                                    drop_last=False, sampler=val_sampler)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    
        fit_one_epoch(
            args=args,
            model_train=model_train,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            epoch_step=epoch_step,
            epoch_step_val=epoch_step_val,
            epoch_step_cam=epoch_step_cam,
            gen=gen,
            gen_val=gen_val,
            cam_gen_val=cam_gen_val,
            metadata_root=os.path.join(args.metadata_root, args.dataset_name, 'val'),
            iou_threshold_list=args.iou_threshold_list,
            dataset_name=args.dataset_name,
            Epoch=args.UnFreeze_Epoch,
            cuda=args.Cuda,
            cam_curve_interval=args.cam_curve_interval,
            eval_period=args.eval_period,
            save_dir=args.save_dir,
            multi_contour_eval=True,
            local_rank=0
        )

    # ---------------------------------------#
    #   最后测试集验证精度
    # ---------------------------------------#
    epoch_step_test = num_test // args.cam_batch_size
    gen_test = DataLoader(test_dataset, shuffle=False, batch_size=args.cam_batch_size, num_workers=num_workers,
                          pin_memory=True,
                          drop_last=False, sampler=val_sampler)
    eval_testset(
        args=args,
        cuda=args.Cuda,
        model_train=model_train,
        gen_test=gen_test,
        metadata_root=os.path.join(args.metadata_root, args.dataset_name, 'test'),
        iou_threshold_list=args.iou_threshold_list,
        dataset_name=args.dataset_name,
        cam_curve_interval=args.cam_curve_interval,
        epoch_step_test=epoch_step_test,
        multi_contour_eval=True,
        save_dir=args.save_dir,
        loss_history=loss_history,
        eval_checkpoint_type=args.eval_checkpoint_type,
        n_samples=args.n_samples,
        perturbations_per_eval=args.perturbations_per_eval,
        local_rank=0,
        split='test',
        backbone=args.backbone,

    )
