import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
from lib import slowfastnet
from tensorboardX import SummaryWriter


def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    train_size=len(train_dataloader.dataset)
    end = time.time()
    for step, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - end
        end = time.time()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if step % params['display'] == 0 and step != 0:
            print('-------------------------------')
            for param in optimizer.param_groups:
                print('lr: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, step, len(train_dataloader))
            print(print_string)
            print_string = 'batch time: {batch_time:.3f} \t'.format(batch_time=batch_time)
            print(print_string)
            ave_loss=running_loss/((step+1)*inputs.size(0))
            print_string = 'Loss {loss:.5f} '.format(loss=ave_loss)
            print(print_string)
            ave_acc = running_corrects.double() /((step+1)*inputs.size(0))
            print_string = 'Average_accuracy {ave_acc:.5f} '.format(ave_acc=ave_acc)
            print(print_string)
    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
    writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)


def validation(model, val_dataloader, epoch, criterion, optimizer, writer):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    val_size=len(val_dataloader.dataset)
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if step % params['display'] == 0 and step != 0:
                print('--validation--')
                print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, step, len(val_dataloader))
                print(print_string)
                ave_loss = running_loss / ((step+1) * inputs.size(0))
                print_string = 'Loss {loss:.5f} '.format(loss=ave_loss)
                print(print_string)
                ave_acc = running_corrects.double() / ((step+1) * inputs.size(0))
                print_string = 'Average_accuracy {ave_acc:.5f} '.format(ave_acc=ave_acc)
                print(print_string)
        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size
        writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
        writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)


def main():
    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")
    train_dataloader = \
        DataLoader(
            VideoDataset(params['dataset'], mode='train', clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate']),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    val_dataloader = \
        DataLoader(
            VideoDataset(params['dataset'], mode='validation', clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate']),
            batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    print("load model")
    model = slowfastnet.resnet50(class_num=params['num_classes'])
    
    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load pretrain model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=0.1)

    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    for epoch in range(params['epoch_num']):
        train(model, train_dataloader, epoch, criterion, optimizer, writer)
        if epoch % 2 == 0:
            validation(model, val_dataloader, epoch, criterion, optimizer, writer)
        scheduler.step()
        if epoch % 1 == 0:
            checkpoint = os.path.join(model_save_dir,
                                      "clip_len_" + str(params['clip_len']) + "frame_sample_rate_" +str(params['frame_sample_rate'])+ "_checkpoint_" + str(epoch) + ".pth.tar")
            torch.save(model.module.state_dict(), checkpoint)

    writer.close

if __name__ == '__main__':
    main()
