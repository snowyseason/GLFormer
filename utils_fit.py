import os
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from utils.utils import get_lr
from get_cam import CAMComputer
import numpy as np


def fit_one_epoch(args, model_train, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                           epoch_step_cam, gen, gen_val, cam_gen_val,
                           metadata_root, iou_threshold_list, dataset_name, Epoch, cuda, 
                           cam_curve_interval, eval_period, save_dir, multi_contour_eval, local_rank=0):
    total_loss = 0
    total_accuracy = 0
    total_accuracy1 = 0
    total_mid_accuracy = 0


    val_loss = 0
    val_accuracy = 0
    val_accuracy1 = 0
    val_mid_accuracy = 0
    

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets, _ = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        # ----------------------#
        #   前向传播
        # ----------------------#
        outputs = model_train(images)

        if len(outputs) == 2:
            outputs, mid_outputs = outputs
            loss_value1 = nn.CrossEntropyLoss()(outputs, targets)
            loss_value2 = nn.CrossEntropyLoss()(mid_outputs, targets)
            loss_value = loss_value1 + loss_value2
            show_flag = 2
        elif len(outputs) == 3:
            x_conv_logit, x_patch_logits, x_cls_logits = outputs
            loss_value1 = nn.CrossEntropyLoss()(x_conv_logit, targets)
            loss_value2 = nn.CrossEntropyLoss()(x_patch_logits, targets)
            loss_value3 = nn.CrossEntropyLoss()(x_cls_logits, targets)
            loss_value = loss_value1 + loss_value2 + loss_value3
            show_flag = 3
        else:
            loss_value = nn.CrossEntropyLoss()(outputs, targets)

        # ----------------------#
        #   反向传播
        # ----------------------#
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()
        
        if show_flag == 1:
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                'acc': total_accuracy / (iteration + 1),
                'lr': get_lr(optimizer)})
            pbar.update(1)
            
        elif show_flag == 2:
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()
                mid_accuracy = torch.mean((torch.argmax(F.softmax(mid_outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_mid_accuracy += mid_accuracy.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                'mid_acc' : total_mid_accuracy / (iteration + 1),
                'acc': total_accuracy / (iteration + 1),
                'lr': get_lr(optimizer)})
            pbar.update(1)

        elif show_flag == 3:
            with torch.no_grad():
                acc_c = torch.mean((torch.argmax(F.softmax(x_conv_logit, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += acc_c.item()
                acc_t_cls = torch.mean((torch.argmax(F.softmax(x_cls_logits, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_mid_accuracy += acc_t_cls.item()
                acc_t_patch = torch.mean((torch.argmax(F.softmax(x_patch_logits, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy1 += acc_t_patch.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                            'acc_c': total_accuracy / (iteration + 1),
                            'acc_t': total_mid_accuracy / (iteration + 1),
                            'acc_p': total_accuracy1 / (iteration + 1),
                            'lr': get_lr(optimizer)})
            pbar.update(1)
    
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
        
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, _ = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()
            
            outputs = model_train(images)
            if len(outputs) == 2:
                outputs, mid_outputs = outputs
                loss_value1 = nn.CrossEntropyLoss()(outputs, targets)
                loss_value2 = nn.CrossEntropyLoss()(mid_outputs, targets)
                loss_value = loss_value1 + loss_value2
            elif len(outputs) == 3:
                x_conv_logit, x_patch_logits, x_cls_logits = outputs
                loss_value1 = nn.CrossEntropyLoss()(x_conv_logit, targets)
                loss_value2 = nn.CrossEntropyLoss()(x_patch_logits, targets)
                loss_value3 = nn.CrossEntropyLoss()(x_cls_logits, targets)
                loss_value = loss_value1 + loss_value2 + loss_value3
            else:
                loss_value = nn.CrossEntropyLoss()(outputs, targets)

        if show_flag == 1:
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy += accuracy.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'acc': val_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
        elif show_flag == 2:
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy += accuracy.item()
            mid_accuracy = torch.mean((torch.argmax(F.softmax(mid_outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_mid_accuracy += mid_accuracy.item()

            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'acc': val_accuracy / (iteration + 1),
                                'mid_acc': val_mid_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
        elif show_flag == 3:
            acc_c = torch.mean((torch.argmax(F.softmax(x_conv_logit, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy += acc_c.item()
            acc_t_cls = torch.mean((torch.argmax(F.softmax(x_cls_logits, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_mid_accuracy += acc_t_cls.item()
            acc_t_patch = torch.mean((torch.argmax(F.softmax(x_patch_logits, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy1 += acc_t_patch.item()

            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                            'acc_c': val_accuracy / (iteration + 1),
                            'acc_t': val_mid_accuracy / (iteration + 1),
                            'acc_p': val_accuracy1 / (iteration + 1),
                            'lr': get_lr(optimizer)})
            pbar.update(1)
        
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        loss_history.append_acc(epoch + 1, total_accuracy / epoch_step, val_accuracy / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        if eval_period != None:
            if (epoch + 1) % eval_period == 0 or epoch + 1 == Epoch:
                print("Start computing and evaluating cams.")
                cam_computer = CAMComputer(
                    model=model_train,
                    loader=cam_gen_val,
                    metadata_root=metadata_root,
                    mask_root='dataset/',
                    iou_threshold_list=iou_threshold_list,
                    dataset_name=dataset_name,
                    split='test',
                    cam_curve_interval=cam_curve_interval,
                    epoch=epoch,
                    epoch_step=epoch_step_cam,
                    Epoch=Epoch,
                    multi_contour_eval=multi_contour_eval,
                    log_folder=save_dir,
                    args=args,
                )
                cam_performance = cam_computer.compute_and_evaluate_cams2()
                MaxBoxAccv2_now = sum(cam_performance) / len(cam_performance)
                print(f'Box30,Box50,Box70,Boxv2:{cam_performance[0], cam_performance[1],cam_performance[2],MaxBoxAccv2_now}')
                loss_history.append_maxboxacc(epoch + 1, cam_performance)
                pbar.close()
                print("Finish computing and evaluating cams.")

                if len(loss_history.MaxBoxAccv2) <= 1 or (MaxBoxAccv2_now) >= max(loss_history.MaxBoxAccv2):
                    print('Save best model to best_maxboxaccv2_weights.pth')
                    print(
                        f'The maxboxacc performance (30,50,70,mean): ({cam_performance[0]}, {cam_performance[1]}, {cam_performance[2]}, {MaxBoxAccv2_now})')
                    torch.save(model_train.state_dict(), os.path.join(save_dir, f"best_maxboxacc_weights.pth"))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        # if save_period != None:
        #     if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        #         torch.save(model_train.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
        #             epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        # if len(loss_history.val_acc) <= 1 or (val_accuracy / epoch_step_val) >= max(loss_history.val_acc):
        #     print('Save best model to best_acc_weights.pth')
        #     torch.save(model_train.state_dict(), os.path.join(save_dir, f"best_acc_weights.pth"))
        #     best_acc_epoch = epoch + 1

        # if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        #     print('Save best model to best_epoch_weights.pth')
        #     torch.save(model_train.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        # report val acc
        max_acc_value = max(loss_history.val_acc)
        max_acc_index = loss_history.val_acc.index(max_acc_value)
        print(f'The best acc is {max_acc_value} located on {max_acc_index}')

        if loss_history.MaxBoxAcc_30 != [] and loss_history.MaxBoxAcc_50 != [] and loss_history.MaxBoxAcc_70 != [] and loss_history.MaxBoxAccv2 != []:
            max_boxacc30_value = max(loss_history.MaxBoxAcc_30)
            max_boxacc30_index = loss_history.MaxBoxAcc_30.index(max_boxacc30_value)
            max_boxacc50_value = max(loss_history.MaxBoxAcc_50)
            max_boxacc50_index = loss_history.MaxBoxAcc_50.index(max_boxacc50_value)
            max_boxacc70_value = max(loss_history.MaxBoxAcc_70)
            max_boxacc70_index = loss_history.MaxBoxAcc_70.index(max_boxacc70_value)
            max_boxaccv2_value = max(loss_history.MaxBoxAccv2)
            max_boxaccv2_index = loss_history.MaxBoxAccv2.index(max_boxaccv2_value)

            print(f"current_max_boxacc:{max}")

            print(
                f'The maxboxacc (30,50,70,mean) is {max_boxacc30_value, max_boxacc50_value, max_boxacc70_value, max_boxaccv2_value} located on '
                f'{max_boxacc30_index, max_boxacc50_index, max_boxacc70_index, max_boxaccv2_index}  ')

        # torch.save(model_train.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))



def eval_testset(args, cuda, model_train, gen_test, metadata_root, iou_threshold_list, dataset_name, cam_curve_interval,
                 epoch_step_test, multi_contour_eval, save_dir, loss_history, eval_checkpoint_type, split, local_rank=0, **kwargs):
    backbone = kwargs['backbone']
    print("Final epoch evaluation on test set ...")

    test_accuracy = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if eval_checkpoint_type == 'best':
        print(f'Load weights from {os.path.join(save_dir, f"best_maxboxacc_weights.pth")}')

        model_dict = model_train.state_dict()
        pretrained_dict = torch.load(os.path.join(save_dir, f"best_maxboxacc_weights.pth"), map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model_train.load_state_dict(model_dict)

        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    print("Start computing and evaluating classfication acc.")
    model_train.eval()
    pbar = tqdm(total=epoch_step_test, postfix=dict, mininterval=0.3)
    for iteration, batch in enumerate(gen_test):
        if iteration >= epoch_step_test:
            break
        images, targets, _ = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            logits = model_train(images)
            if len(logits) == 3:
                x_conv_logit, x_patch_logits, x_cls_logits = logits
                outputs = (x_conv_logit + x_patch_logits + x_cls_logits) / 3
            elif len(logits) == 2:
                x_patch_logits, x_cls_logits = logits
                outputs = (x_patch_logits + x_cls_logits) / 3

            
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
        test_accuracy += accuracy.item()

        pbar.set_postfix(**{
            'accuracy': test_accuracy / (iteration + 1),
        })
        pbar.update(1)
    pbar.close()
    test_acc = test_accuracy / epoch_step_test
    print(f'Top1 test accuracy: {test_acc}')
    print('Finish computing accuracy')
    loss_history.write_testacc(test_acc, split)

    print("Start computing and evaluating cams.")

    cam_computer = CAMComputer(
        model=model_train,
        loader=gen_test,
        metadata_root=metadata_root,
        mask_root='dataset/',
        iou_threshold_list=iou_threshold_list,
        dataset_name=dataset_name,
        split='test',
        cam_curve_interval=cam_curve_interval,
        epoch=None,
        epoch_step=epoch_step_test,
        Epoch=None,
        multi_contour_eval=multi_contour_eval,
        log_folder=save_dir,
        backbone=backbone,
        args = args,
    )

    cam_performance = cam_computer.compute_and_evaluate_cams2()
    print(f'The testset performance: {cam_performance}')
    loss_history.plotbar_boxacc(cam_performance, split)
    print("Finish computing and evaluating cams.")







