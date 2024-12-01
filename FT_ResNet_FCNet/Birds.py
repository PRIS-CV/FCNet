import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

import torch
from torch.autograd import Variable
import json
from collections import defaultdict
from utils import *
from configs import *
from network import *
from torch.autograd import Variable
from torchvision.datasets.folder import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parser = argparse.ArgumentParser(description='Integrate FCNet into ResNet-50 on CUB-200-2011')

parser.add_argument('--data_url', default='/mnt/sdb/data/chenjunhan/Cross-Granularity/Birds/', type=str,
                    help='path to the dataset (CUB-200-2011)')

parser.add_argument('--work_dirs', default='/home/chenjunhan/FCNet/FT_ResNet_FCNet/weights', type=str,
                    help='path to save log and checkpoints')

parser.add_argument('--train_stage', default=1, type=int,
                    help='select training stage')        

parser.add_argument('--checkpoint_path', default='/home/chenjunhan/FCNet/FCNet_Training/weights/RL_train-stage2/model_best.pth.tar', type=str,
                    help='path to the stage-2/3 checkpoint (for training stage-2/3)')

args = parser.parse_args()


def main():

    if not os.path.isdir(args.work_dirs):
        mkdir_p(args.work_dirs)

    record_path = args.work_dirs + '/RL' + '_train-stage' + str(args.train_stage)
    if not os.path.isdir(record_path):
        mkdir_p(record_path)
    
    model_configurations = model_configuration['resnet50']
    train_configurations = train_configuration['resnet50']

    model = models.resnet.resnet18(pretrained=False)
    model = model_ResNet_Crop(model)
    model_prime = models.resnet.resnet50(pretrained=False)
    model_prime = model_ResNet(model_prime)

    fc = Crop_layer(model_configurations['crop_num'], train_configurations['classes_num'])
    
    fc_prime = Full_layer(model_configurations['feature_num'], model_configurations['feature_size'],
                          train_configurations['classes_num'])

    learning_rate = train_configurations['learning_rate']
    checkpoint = torch.load(args.checkpoint_path)

    checkpoint_model = checkpoint['model']
    model_dict = {}
    for k, v in checkpoint_model.items():
        new_key = k.replace('backbone.module.','backbone.')
        model_dict[new_key] = v
    model.load_state_dict(model_dict)

    checkpoint_fc = checkpoint['fc']
    fc_dict = {}
    for k, v in checkpoint_fc.items():
        new_key = k.replace('module.','')
        fc_dict[new_key] = v
    fc.load_state_dict(fc_dict)

    optimizer = torch.optim.SGD([
        {'params': model_prime.backbone.parameters(), "lr": learning_rate / 10},
        {'params': fc_prime.features.parameters(), "lr": learning_rate},
        {'params': fc_prime.classifier.parameters(), "lr": learning_rate}
        ], momentum=train_configurations['momentum'], weight_decay=train_configurations['weight_decay'])
    
    training_epoch_num = train_configurations['epoch_num']    
    criterion = nn.CrossEntropyLoss()

    model.backbone.cuda()
    model_prime.backbone.cuda()

    fc.features.cuda()
    fc.classifier.cuda()
    fc_prime.features.cuda()
    fc_prime.classifier.cuda()

    model.backbone = torch.nn.DataParallel(model.backbone)
    model_prime.backbone = torch.nn.DataParallel(model_prime.backbone)

    fc.features = torch.nn.DataParallel(fc.features)
    fc.classifier = torch.nn.DataParallel(fc.classifier)
    fc_prime.features = torch.nn.DataParallel(fc_prime.features)
    fc_prime.classifier = torch.nn.DataParallel(fc_prime.classifier)

    train_dir = args.data_url + 'train/'
    val_dir = args.data_url + 'test/'

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_train_crop = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_val_crop = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    train_set = MyImageFolder(root=train_dir, transform=transform_train, transform_crop = transform_train_crop)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_configurations['batch_size'], shuffle=True,
                                               num_workers=train_configurations['num_workers'], drop_last=True)
    
    val_set = MyImageFolder(root=val_dir, transform=transform_val, transform_crop = transform_val_crop)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=train_configurations['batch_size'], shuffle=True,
                                               num_workers=train_configurations['num_workers'], drop_last=True)

    state_dim = model_configurations['feature_map_channels'] * math.ceil(train_configurations['image_size'] / 32) * math.ceil(train_configurations['image_size'] / 32)
    ppo = PPO(model_configurations['feature_map_channels'], state_dim, model_configurations['policy_hidden_dim'])

    ppo.policy.load_state_dict(checkpoint['policy'])
    ppo.policy_old.load_state_dict(checkpoint['policy'])
    for param in model.parameters():
            param.requires_grad = False

    memory = Memory()

    start_epoch = 0
    best_acc_prime = 0
    

    for epoch in range(start_epoch, training_epoch_num):

        print('Training Stage: {}, lr:'.format(args.train_stage))
        adjust_learning_rate(optimizer, train_configurations, epoch, training_epoch_num, args)

        train(model, model_prime, fc, fc_prime, memory, ppo, optimizer, train_set, train_loader, 
              criterion, epoch, args)
        
        acc_prime = validate(model, model_prime, fc, fc_prime, memory, ppo, val_set, val_loader,
                 criterion, epoch, args, best_acc_prime)
    
        if acc_prime > best_acc_prime:
            best_acc_prime = acc_prime
            acc_prime_best = True
        else:
            acc_prime_best = False
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'fc': fc.state_dict(),
            'policy': ppo.policy.state_dict() if ppo else None,
        }, acc_prime_best, checkpoint=record_path)

        save_prime({
            'model_prime': model_prime.state_dict(),
            'fc_prime': fc_prime.state_dict(),
            'acc_prime': acc_prime,
            'best_acc_prime': best_acc_prime,
        }, acc_prime_best, checkpoint=record_path)


def train(model, model_prime, fc, fc_prime, memory, ppo, optimizer, 
          train_set, train_loader, criterion, epoch, args):

    total = 0
    correct = 0
    total_prime = 0
    correct_prime = 0

    loss_total = 0
    train_loss = 0

    model.train()
    model_prime.train()
    fc.train()
    fc_prime.train()

    for i, (inputs, targets, index) in enumerate(train_loader):

        idx = i

        inputs_var = inputs.cuda()
        targets_var = targets.cuda()

        inputs_var, targets_var = Variable(inputs_var), Variable(targets_var)

        output, state = model(inputs_var)
        output = fc(output)

        _, predicted = torch.max(output.data, 1)
        total += targets_var.size(0)
        correct += predicted.eq(targets_var.data).cpu().sum().item()

        action = ppo.select_action(state, memory)

        patches = train_set.get_patch(index, action).cuda()

        output, state = model_prime(patches)
        output = fc_prime(output) 

        loss_prime = criterion(output, targets_var)

        _, predicted_prime = torch.max(output.data, 1)
        total_prime += targets_var.size(0)
        correct_prime += predicted_prime.eq(targets_var.data).cpu().sum().item()
        
        loss_total = loss_prime

        train_loss += loss_total.item()

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    acc = 100.* correct / total
    acc_prime = 100.* correct_prime / total_prime
    train_loss = train_loss / (idx + 1)
    print('Iteration %d, train_acc = %.5f, train_acc_prime = %.5f, train_loss = %.6f' %\
              (epoch, acc, acc_prime, train_loss))
    

def validate(model, model_prime, fc, fc_prime, memory, ppo,
             val_set, val_loader, criterion, epoch, args, best_acc_prime):
    
    total = 0
    correct = 0
    total_prime = 0
    correct_prime = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    loss_total = 0
    val_loss = 0

    model.eval()
    model_prime.eval()
    fc.eval()
    fc_prime.eval()

    with torch.no_grad():
        for i, (inputs, targets, index) in enumerate(val_loader):

            idx = i
            loss_cla = []

            inputs_var = inputs.cuda()
            targets_var = targets.cuda()
            inputs_var, targets_var = Variable(inputs_var), Variable(targets_var)

            output, state = model(inputs_var)
            output = fc(output)

            _, predicted = torch.max(output.data, 1)
            total += targets_var.size(0)
            correct += predicted.eq(targets_var.data).cpu().sum().item()

            action = ppo.select_action(state, memory, training=False) 
            patches = val_set.get_patch(index, action).cuda()

            output, state = model_prime(patches)
            output = fc_prime(output) 

            loss_prime = criterion(output, targets_var)
            loss_cla.append(loss_prime)

            _, predicted_prime = torch.max(output.data, 1)
            total_prime += targets_var.size(0)
            correct_prime += predicted_prime.eq(targets_var.data).cpu().sum().item()

            loss_total = sum(loss_cla)
            val_loss += loss_total.item()

    acc = 100. * correct / total
    acc_prime = 100. * correct_prime / total_prime
    val_loss = val_loss / (idx + 1)

    print('Iteration %d, test_acc = %.5f, test_acc_prime = %.5f, test_loss = %.6f' % \
          (epoch, acc, acc_prime, val_loss))

    return acc_prime


if __name__ == '__main__':
    main()
        
        
