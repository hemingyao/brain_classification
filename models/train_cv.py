from setting import parse_opts 
from datasets.dataset_loader import OCTDataset
import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import time
from models.train_utils.logger import log
from models.train_utils.avg_meter import avg_meter
import models.train_utils.metrics as metrics
from models.train_utils.imbalanced import ImbalancedDatasetSampler
from models.train_utils.loss import focal_loss
import random
import os
import model


def train(data_loader, val_data_loader, net, optimizer, scheduler, criterion, total_epochs, save_interval, save_folder, sets, fold):
    # settings

    if sets.vistool == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(sets.tensorboard_dir + f'_fold{fold}')#default log dir, will create folder
        print('Tensorboard log at '+sets.tensorboard_dir)

    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

    print("Current setting is:")
    print(sets)
    print("\n\n")     

    net.train()

    losses = avg_meter()
    accuracies = avg_meter()

    train_time_sp = time.time()
    running_loss = []
    running_accuracy = []
    record_steps = 10
    
    best_val_sum_aucpr = 0
    best_epoch = 0
    best_val_metrics_mat = None
    patience = 0
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        log.info('lr = {}'.format(scheduler.get_lr()))
        
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, targets = batch_data
            if '2D' in sets.target_label:
                volumes = torch.reshape(volumes, (-1,) + volumes.shape[-3:])
                targets = torch.reshape(targets, (-1,))
            
            if not sets.no_cuda: 
                volumes = volumes.cuda()
                targets = targets.cuda()

            outputs = net(volumes)
            loss = criterion(outputs, targets)
            acc = metrics.calculate_accuracy(outputs, targets)

            losses.update(loss.item(), volumes.size(0))
            accuracies.update(acc, volumes.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            
            running_loss.append(losses.val)
            running_accuracy.append(accuracies.val)
            
            if sets.vistool == 'tensorboard' and (batch_id_sp+batch_id)%record_steps == 0:
                loss_mean = np.mean(running_loss)
                acc_mean = np.mean(running_accuracy)
                running_loss = []
                running_accuracy = []
                
                writer.add_scalar(f'training_loss', loss_mean, batch_id_sp+batch_id)
                writer.add_scalar(f'training_accuracy', acc_mean, batch_id_sp+batch_id)
                log.info(
                    'Fold {}, Batch: {}-{} ({}), loss = {:.3f}, acc = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(fold, epoch, batch_id, batch_id_sp+batch_id, loss_mean, acc_mean, avg_batch_time))
            
            if not sets.ci_test:
                # eval and save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    net.eval() # switch to evalation mode 
                    with torch.no_grad():
                        targets = torch.tensor(())
                        probs = torch.tensor(())

                        step = 0
                        for batch_id, batch_data in enumerate(val_data_loader):
                            step += 1

                            # forward
                            volume, target = batch_data
                            if '2D' in sets.target_label:
                                volume = torch.reshape(volume, (-1,) + volume.shape[-3:])
                                target = torch.reshape(target, (-1,))

                            if not sets.no_cuda:
                                volume = volume.cuda()
                            output = net(volume)    
                            prob = torch.nn.functional.softmax(output, dim=-1)

                            targets = torch.cat((targets, target.float()), 0)
                            probs = torch.cat((probs, prob.cpu()), 0)

                        val_acc = metrics.calculate_accuracy(probs, targets)
                        val_metrics, metrics_names = metrics.calculate_metrics_for_individual_class(probs, targets, n_classes = sets.n_classes,
                                                                                                    label_list=list(range(sets.n_classes))[1:])

                    log_string = f"Fold{fold}: Validation accuracy = {val_acc:.3f}"
                    tensorb_scalar_dict = {f'Accuracy': val_acc}

                    val_metrics_list = []
                    sum_aucpr = 0
                    for label_name in val_metrics:
                        metrics_array = val_metrics[label_name]
                        val_metrics_list.append(metrics_array)
                        log_string += f', AUC_class_{label_name} = {metrics_array[-1]:.3f}'
                        log_string += f', AUCPR_class_{label_name} = {metrics_array[-2]:.3f}'
                        sum_aucpr += metrics_array[-2]
                        tensorb_scalar_dict[f'AUC_class_{label_name}'] = metrics_array[-1]
                        tensorb_scalar_dict[f'AUCPR_class_{label_name}'] = metrics_array[-2]
                        if sets.use_spell:
                            spell.metrics.send_metric(f"Fold{fold}_val_AUC_class_{label_name}", metrics_array[-1])
                            spell.metrics.send_metric(f"Fold{fold}_val_AUCPR_class_{label_name}", metrics_array[-2])

                    log.info(log_string)
                    if sets.vistool == 'tensorboard':
                        writer.add_scalars('validation', tensorb_scalar_dict, batch_id_sp+batch_id)

                        if sets.n_classes == 2:
                            writer.add_figure(f'ROC', metrics.plot_ROC(probs, targets), batch_id_sp+batch_id)
                            writer.add_figure(f'PRC', metrics.plot_Recall_Precision_Curve(probs, targets), batch_id_sp+batch_id)

                        writer.add_figure('Confusion_matrix', metrics.plot_confusion_matrix(probs, targets, sets.n_classes), batch_id_sp+batch_id)

                    
                    # Save the best val performance
                    if sum_aucpr > best_val_sum_aucpr:
                        patience = 0
                        best_val_sum_aucpr = sum_aucpr
                        best_epoch = epoch
                        best_val_metrics_mat = np.stack(val_metrics_list, axis=0)

                        # Save this epoch
                        model_save_path = '{}_fold_{}_best_epoch.pth.tar'.format(save_folder, fold, epoch, batch_id)
                        model_save_dir = os.path.dirname(model_save_path)
                        if not os.path.exists(model_save_dir):
                            os.makedirs(model_save_dir)
                        log.info('Save best checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                        torch.save({
                                    'ecpoch': epoch,
                                    'batch_id': batch_id,
                                    'state_dict': net.state_dict(),
                                    'optimizer': optimizer.state_dict()},
                                    model_save_path)
                    else:
                        patience += 1
                        
                    if batch_id_sp % (5*save_interval) == 0:
                        model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                        model_save_dir = os.path.dirname(model_save_path)
                        if not os.path.exists(model_save_dir):
                            os.makedirs(model_save_dir)
                        log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                        torch.save({
                                    'ecpoch': epoch,
                                    'batch_id': batch_id,
                                    'state_dict': net.state_dict(),
                                    'optimizer': optimizer.state_dict()},
                                    model_save_path)

                    net.train() # switch back to train mode
                    
                    if patience > 20:
                        break
                        
        scheduler.step()
                            
    print(f'Finished training for fold {fold}. The best epoch is {best_epoch}.')
    
    if sets.ci_test:
        exit()
    return best_val_metrics_mat


def build_model_cv(sets, fold):
    device = 'cpu' if sets.no_cuda else 'cuda'
    
    # getting model
    if sets.network == '2D-resnet':
        net = model.ResNet(n_classes=sets.n_classes, add_attention=sets.add_attention)
    elif sets.network == 'late-fusion':
        net = model.LateFusion(n_classes=sets.n_classes)
        
        if sets.pretrained:
            print('Loading the pretrained model.')
            checkpoint = torch.load('model/nGA_2d_baseline_epoch_20_batch_311.pth.tar', map_location=torch.device(device))
            from collections import OrderedDict
            updated_state_dict = OrderedDict()
            for i in checkpoint['state_dict']:
                if i != 'fc1.weight' and i != 'fc1.bias':
                    updated_state_dict[i] = checkpoint['state_dict'][i]

            net.load_state_dict(updated_state_dict, strict=False)
            # #net.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print('Using the ImageNet pretraiend model.')

    else:
        raise NotImplementedError(sets.network + ' has not been implemented.')
        
    net.to(device)
    parameters = net.parameters()
    
    print(net)

    if sets.nesterov:
        dampening = 0
    else:
        dampening = sets.dampening
        
    optimizer = SGD(parameters,
                    lr = sets.learning_rate,
                    momentum=sets.momentum,
                    dampening=dampening,
                    weight_decay=sets.weight_decay,
                    nesterov=sets.nesterov)

    assert sets.lr_scheduler in ['plateau', 'multistep']
    assert not (sets.lr_scheduler == 'plateau' and sets.no_val)
    if sets.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=sets.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             sets.multistep_milestones)

    # train from resume
    #TODO check setting for sets.multistep_milestones
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    
    if sets.loss == 'cross_entropy':
        # TODO: a better way for multi-classification
        criterion = CrossEntropyLoss(torch.FloatTensor([1] + [sets.pos_weight]*(sets.n_classes-1)))
        if not sets.no_cuda:
            criterion = criterion.cuda()
    elif sets.loss == 'focal_loss':
        criterion = focal_loss(alpha=sets.focal_loss_alpha, gamma=sets.focal_loss_gamma, reduction='mean', device=device)

    print('Build a model with fold {}'.format(fold))
    training_dataset = OCTDataset(sets.image_path_cera, sets.label_file, sets.split_path, 
                                      image_path_proximab=sets.image_path_proximab,
                                      image_path_chroma=sets.image_path_chroma,
                                      image_set='train',target_label = sets.target_label, transform=sets.augmentation, 
                                      shuffle=True, cv_fold=fold)
    
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, 
                             sampler=ImbalancedDatasetSampler(training_dataset, callback_get_label=None),#train_dataset.get_label
                             num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    validation_dataset = OCTDataset(sets.image_path_cera, sets.label_file, sets.split_path, 
                                        image_path_proximab=sets.image_path_proximab,
                                        image_path_chroma=sets.image_path_chroma,
                                        image_set='val', target_label = sets.target_label, cv_fold=fold)
    
    val_data_loader = DataLoader(validation_dataset, batch_size=1, 
                                 num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    best_val_metrics_mat = train(data_loader, val_data_loader, net, optimizer, scheduler, criterion, 
                                 total_epochs=sets.n_epochs, 
                                 save_interval=sets.save_intervals, 
                                 save_folder=sets.save_folder, sets=sets, fold=fold) 
    return best_val_metrics_mat

    
if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    
    if sets.use_spell:
        import spell.metrics
        
    # set random seed for torch, numpy and random
    torch.manual_seed(sets.manual_seed)
    np.random.seed(sets.manual_seed)
    random.seed(sets.manual_seed)
    
    num_folds = 5
    fold_eval_list = []
    for ind in range(num_folds):
        eval_mat = build_model_cv(sets, fold=ind)
        print('Fold ', ind)
        print(eval_mat)
        fold_eval_list.append(eval_mat)
    
    fold_eval_mat = np.stack(fold_eval_list, axis=0)
    print(fold_eval_mat)
    np.save('fold_eval_mat.npy', fold_eval_mat)
