import sys
sys.path.append("../")

from setting import parse_opts 
from datasets.dataset_loader import BrainDataset
import torch
import numpy as np
from torch.optim import Adam
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


def train(data_loader, val_data_loader, net, optimizer, criterion, total_epochs, save_interval, save_folder, sets):
    # settings
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(sets.tensorboard_dir)#default log dir, will create folder
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
    global_step = 0

    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            global_step += 1
            images, targets = batch_data
            
            if not sets.no_cuda: 
                images = images.cuda()
                targets = targets.cuda()

            outputs = net(images)
            loss = criterion(outputs, targets)
            acc = metrics.calculate_accuracy(outputs, targets)

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            
            running_loss.append(losses.val)
            running_accuracy.append(accuracies.val)
            
            if global_step%record_steps == 0:
                loss_mean = np.mean(running_loss)
                acc_mean = np.mean(running_accuracy)
                running_loss = []
                running_accuracy = []
                
                writer.add_scalar('training_loss', loss_mean, global_step)
                writer.add_scalar('training_accuracy', acc_mean, global_step)
                log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, acc = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, global_step, loss_mean, acc_mean, avg_batch_time))
            
            # eval and save model
            if batch_id == 0 and global_step != 0 and global_step % save_interval == 0:
                net.eval() # switch to evalation mode 
                with torch.no_grad():
                    targets = torch.tensor(())
                    probs = torch.tensor(())

                    step = 0
                    for batch_id, batch_data in enumerate(val_data_loader):
                        step += 1

                        # forward
                        images, target = batch_data
                        images = torch.reshape(images, (-1,) + images.shape[-3:])
                        target = torch.reshape(target, (-1,))

                        if not sets.no_cuda:
                            images = images.cuda()
                        output = net(images)    
                        prob = torch.nn.functional.softmax(output, dim=-1)

                        targets = torch.cat((targets, target.float()), 0)
                        probs = torch.cat((probs, prob.cpu()), 0)

                    val_acc = metrics.calculate_accuracy(probs, targets)
                    val_metrics, metrics_names = metrics.calculate_metrics_for_individual_class(probs, targets, n_classes = sets.n_classes,
                                                                                                label_list=list(range(sets.n_classes))[1:])

                log_string = f"Validation accuracy = {val_acc:.3f}"
                tensorb_scalar_dict = {'Accuracy': val_acc}
                sum_aucpr = 0
                val_metrics_list = []
                for label_name in val_metrics:
                    metrics_array = val_metrics[label_name]
                    val_metrics_list.append(metrics_array)
                    log_string += f', AUC_class_{label_name} = {metrics_array[-1]:.3f}'
                    log_string += f', AUCPR_class_{label_name} = {metrics_array[-2]:.3f}'
                    sum_aucpr += metrics_array[-2]
                    tensorb_scalar_dict[f'AUC_class_{label_name}'] = metrics_array[-1]
                    tensorb_scalar_dict[f'AUCPR_class_{label_name}'] = metrics_array[-2]

                log.info(log_string)

                writer.add_scalars('validation', tensorb_scalar_dict, batch_id_sp+batch_id)

                if sets.n_classes == 2:
                    writer.add_figure('ROC', metrics.plot_ROC(probs, targets), batch_id_sp+batch_id)
                    writer.add_figure('PRC', metrics.plot_Recall_Precision_Curve(probs, targets), batch_id_sp+batch_id)

                writer.add_figure('Confusion_matrix', metrics.plot_confusion_matrix(probs, targets, sets.n_classes), batch_id_sp+batch_id)

                # Save the best val performance
                if sum_aucpr > best_val_sum_aucpr:
                    patience = 0
                    best_val_sum_aucpr = sum_aucpr
                    best_epoch = epoch
                    best_val_metrics_mat = np.stack(val_metrics_list, axis=0)

                    # Save this epoch
                    model_save_path = '{}_best_epoch.pth.tar'.format(save_folder, epoch, batch_id)
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

                            
    print('Finished training')            

    print('The best epoch is {}'.format(best_epoch))
    print('Best validation accuracy is ', best_val_metrics_mat)
    return best_val_metrics_mat


def build_model(sets):
    image_size = (512, 512)
    device = 'cpu' if sets.no_cuda else 'cuda'
    
    # getting model
    net = model.Network(architecture=sets.network, 
                        n_classes=sets.n_classes, 
                        pretrained=sets.pretrained)

    net.to(device)
    parameters = net.parameters()
    
    print(net)
    
    optimizer = Adam(parameters, lr = sets.learning_rate)

    # getting data
    if sets.loss == 'cross_entropy':
        # TODO: a better way for multi-classification
        criterion = CrossEntropyLoss()
        if not sets.no_cuda:
            criterion = criterion.cuda()

    elif sets.loss == 'focal_loss':
        criterion = focal_loss(alpha=sets.focal_loss_alpha, gamma=sets.focal_loss_gamma, reduction='mean', device=device)

    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    

    training_dataset = BrainDataset(image_set=sets.image_set + '_train.csv', 
                                    image_size=image_size, augmentation=sets.augmentation)
    
    train_data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, 
                             sampler=ImbalancedDatasetSampler(training_dataset, callback_get_label=None),
                             num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    validation_dataset = BrainDataset(image_set=sets.image_set + '_val.csv', 
                                    image_size=image_size, augmentation=sets.augmentation)
    
    val_data_loader = DataLoader(validation_dataset, batch_size=1, 
                                 num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    train(train_data_loader, val_data_loader, net, optimizer, criterion, 
          total_epochs=sets.n_epochs, 
          save_interval=sets.save_intervals, 
          save_folder=sets.save_folder, sets=sets) 
    
    
if __name__ == '__main__':
    # settting
    sets = parse_opts()   
        
    # set random seed for torch, numpy and random
    torch.manual_seed(sets.manual_seed)
    np.random.seed(sets.manual_seed)
    random.seed(sets.manual_seed)
    build_model(sets)