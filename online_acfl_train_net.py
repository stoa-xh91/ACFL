#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from numpy.lib.arraysetops import isin
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=16,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--cfml_loss', default=None, help='the model will be used')
    parser.add_argument('--source_sform_model', default=None, help='the model will be used')
    parser.add_argument('--source_mform_model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.best_joint_acc = 0
        self.best_joint_acc_epoch = 0
        self.best_bone_acc = 0
        self.best_bone_acc_epoch = 0
        self.best_hybrid_acc = 0
        self.best_hybrid_acc_epoch = 0

        
        self.joint_model = self.joint_model.cuda(self.output_device)
        self.bone_model = self.bone_model.cuda(self.output_device)
        self.hybrid_model = self.hybrid_model.cuda(self.output_device)
        self.j_CFMLLoss = self.j_CFMLLoss.cuda(self.output_device)
        self.b_CFMLLoss = self.b_CFMLLoss.cuda(self.output_device)
        self.h_CFMLLoss = self.h_CFMLLoss.cuda(self.output_device)
        if type(self.arg.device) is list:

            if len(self.arg.device) > 1:
                self.joint_model = nn.DataParallel(
                    self.joint_model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
                self.bone_model = nn.DataParallel(
                    self.bone_model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
                self.hybrid_model = nn.DataParallel(
                    self.hybrid_model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        UniModel = import_class(self.arg.source_sform_model)
        MultiModel = import_class(self.arg.source_mform_model)
        # shutil.copy2(inspect.getfile(UniModel), self.arg.work_dir)
        self.joint_model = UniModel(**self.arg.model_args)
        
        self.bone_model = UniModel(**self.arg.model_args)
        
        self.hybrid_model = MultiModel(**self.arg.model_args)
        
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        loss_factory = import_class(self.arg.cfml_loss)
        self.j_CFMLLoss = loss_factory(self.arg.model_args['base_channel']*4)
        self.b_CFMLLoss = loss_factory(self.arg.model_args['base_channel']*4)
        self.h_CFMLLoss = loss_factory(self.arg.model_args['base_channel']*4)


    def load_optimizer(self):

        params = []

        for p in self.joint_model.parameters():
            params.append(p)
        for p in self.bone_model.parameters():
            params.append(p)
        for p in self.hybrid_model.parameters():
            params.append(p)

        for p in self.j_CFMLLoss.parameters():
            params.append(p)
        for p in self.b_CFMLLoss.parameters():
            params.append(p)
        for p in self.h_CFMLLoss.parameters():
            params.append(p)
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
                print('learning rate:',lr,self.arg.base_lr)
            else:
                lr = self.arg.base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.joint_model.train()
        self.bone_model.train()
        self.hybrid_model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        d_loss_value = []
        d_feat_loss_value = []
        d_logit_loss_value = []
        acc_value = []

        loss_value_joint = []
        d_loss_value_joint = []
        d_feat_loss_value_joint = []
        d_logit_loss_value_joint = []
        acc_value_joint = []

        loss_value_bone = []
        d_loss_value_bone = []
        d_feat_loss_value_bone = []
        d_logit_loss_value_bone = []
        acc_value_bone = []

        loss_value_hybrid = []
        d_loss_value_hybrid = []
        d_feat_loss_value_hybrid = []
        d_logit_loss_value_hybrid = []
        acc_value_hybrid = []
        j_factor = 0.
        b_factor = 0.
        h_factor = 0.

        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                joint_data = data[:, :3, ...]
                bone_data = data[:, 3:, ...]
            timer['dataloader'] += self.split_time()
            
            jt_output = self.joint_model(joint_data)
            bt_output = self.bone_model(bone_data)
            mm_output = self.hybrid_model(data)

            with torch.no_grad():
                j_logits, b_logits, m_logits = jt_output['logits'].clone(), bt_output['logits'].clone(), mm_output['logits'].clone()
                j_value, j_predict_label = torch.max(j_logits.data, 1)
                j_factor = torch.mean((j_predict_label == label.data).float())
                b_value, b_predict_label = torch.max(b_logits.data, 1)
                b_factor = torch.mean((b_predict_label == label.data).float())
                m_value, m_predict_label = torch.max(m_logits.data, 1)
                h_factor = torch.mean((m_predict_label == label.data).float())
                beta = torch.cat([j_factor.reshape(1, 1, 1), b_factor.reshape(1, 1, 1), h_factor.reshape(1, 1, 1)], dim=2)
                if np.random.uniform(0,1)>0.5:
                    j_old_value = j_logits.data[torch.arange(0, label.size(0)).long(), label.long()]
                    j_logits[torch.arange(0, label.size(0)).long(), label.long()] = 1.* j_value
                    j_logits[torch.arange(0, label.size(0)).long(), j_predict_label.long()] = 1.* j_old_value
                if np.random.uniform(0,1)>0.5:
                    b_old_value = b_logits.data[torch.arange(0, label.size(0)).long(), label.long()]
                    b_logits[torch.arange(0, label.size(0)).long(), label.long()] = 1.* b_value
                    b_logits[torch.arange(0, label.size(0)).long(), b_predict_label.long()] = 1.* b_old_value
                if np.random.uniform(0,1)>0.5:
                    m_old_value = m_logits.data[torch.arange(0, label.size(0)).long(), label.long()]
                    m_logits[torch.arange(0, label.size(0)).long(), label.long()] = 1.* m_value
                    m_logits[torch.arange(0, label.size(0)).long(), m_predict_label.long()] = 1.* m_old_value

                logits_t  = torch.cat([j_logits.unsqueeze(0), b_logits.unsqueeze(0), m_logits.unsqueeze(0)], dim=0)
                feature_t = torch.cat([jt_output['feature'].unsqueeze(0), bt_output['feature'].unsqueeze(0), mm_output['feature'].unsqueeze(0)], dim=0)
                

            loss = 0
            
            j_loss = self.loss(jt_output['logits'], label)
            b_loss = self.loss(bt_output['logits'], label)
            mm_loss = self.loss(mm_output['logits'], label)
            cls_loss = j_loss + b_loss + mm_loss
            
            j_cfml_loss = self.j_CFMLLoss(feature_t, jt_output['feature'].unsqueeze(0), logits_t, jt_output['logits'].unsqueeze(0), beta)
            b_cfml_loss= self.b_CFMLLoss(feature_t, bt_output['feature'].unsqueeze(0), logits_t, bt_output['logits'].unsqueeze(0), beta)
            h_cfml_loss = self.h_CFMLLoss(feature_t, mm_output['feature'].unsqueeze(0), logits_t, mm_output['logits'].unsqueeze(0), beta)
            cfml_loss = j_factor * j_cfml_loss['d_loss'].mean() + b_factor * b_cfml_loss['d_loss'].mean() + h_factor * h_cfml_loss['d_loss'].mean()

            loss = cls_loss + cfml_loss
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()         

            # overall 
            loss = (j_loss + b_loss + mm_loss) * 0.3
            cfml_loss = (j_cfml_loss['d_loss'].mean() + b_cfml_loss['d_loss'].mean() + h_cfml_loss['d_loss'].mean()) * 0.3
            d_feat_loss = (j_cfml_loss['d_feat_loss'].mean() + b_cfml_loss['d_feat_loss'].mean() + h_cfml_loss['d_feat_loss'].mean()) * 0.3
            d_logit_loss = (j_cfml_loss['d_logit_loss'].mean() + b_cfml_loss['d_logit_loss'].mean() + h_cfml_loss['d_logit_loss'].mean()) * 0.3
            output = jt_output['logits'] + bt_output['logits'] + mm_output['logits']


            loss_value.append(loss.data.item())
            d_loss_value.append(cfml_loss.data.item())
            d_feat_loss_value.append(d_feat_loss.data.item())
            d_logit_loss_value.append(d_logit_loss.data.item())

            loss_value_joint.append(j_loss.data.item())
            d_loss_value_joint.append(j_cfml_loss['d_loss'].mean().data.item())
            d_feat_loss_value_joint.append(j_cfml_loss['d_logit_loss'].mean() .data.item())
            d_logit_loss_value_joint.append(j_cfml_loss['d_logit_loss'].mean().data.item())

            loss_value_bone.append(b_loss.data.item())
            d_loss_value_bone.append(b_cfml_loss['d_loss'].mean().data.item())
            d_feat_loss_value_bone.append(b_cfml_loss['d_logit_loss'].mean() .data.item())
            d_logit_loss_value_bone.append(b_cfml_loss['d_logit_loss'].mean().data.item())

            loss_value_hybrid.append(mm_loss.data.item())
            d_loss_value_hybrid.append(h_cfml_loss['d_loss'].mean().data.item())
            d_feat_loss_value_hybrid.append(h_cfml_loss['d_logit_loss'].mean() .data.item())
            d_logit_loss_value_hybrid.append(h_cfml_loss['d_logit_loss'].mean().data.item())

            timer['model'] += self.split_time()
            
            value, predict_label = torch.max(jt_output['logits'].data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value_joint.append(acc.data.item())
            
            
            value, predict_label = torch.max(bt_output['logits'].data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value_bone.append(acc.data.item())
                        
            value, predict_label = torch.max(mm_output['logits'].data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value_hybrid.append(acc.data.item())
            
            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()
        
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        log_msg = '\tOverall: Mean training loss: {:.4f}. Mean distill loss: {:.4f} d_feat {:.4f} d_logit {:.4f}. Mean training acc: {:.2f}%.'
        log_msg += '\n\tJoint model: Mean training loss: {:.4f}. Mean distill loss: {:.4f} d_feat {:.4f} d_logit {:.4f}. Mean training acc: {:.2f}%.'
        log_msg += '\n\tBone model: Mean training loss: {:.4f}. Mean distill loss: {:.4f} d_feat {:.4f} d_logit {:.4f}. Mean training acc: {:.2f}%.'
        log_msg += '\n\tHybrid model: Mean training loss: {:.4f}. Mean distill loss: {:.4f} d_feat {:.4f} d_logit {:.4f}. Mean training acc: {:.2f}%.'
        self.print_log(
            log_msg.format(np.mean(loss_value), np.mean(d_loss_value), np.mean(d_feat_loss_value), np.mean(d_logit_loss_value), np.mean(acc_value)*100,
            np.mean(loss_value_joint), np.mean(d_loss_value_joint), np.mean(d_feat_loss_value_joint), np.mean(d_logit_loss_value_joint), np.mean(acc_value_joint)*100,
            np.mean(loss_value_bone), np.mean(d_loss_value_bone), np.mean(d_feat_loss_value_bone), np.mean(d_logit_loss_value_bone), np.mean(acc_value_bone)*100,
            np.mean(loss_value_hybrid), np.mean(d_loss_value_hybrid), np.mean(d_feat_loss_value_hybrid), np.mean(d_logit_loss_value_hybrid), np.mean(acc_value_hybrid)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.joint_model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-JGCN-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

            state_dict = self.bone_model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-BGCN-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

            state_dict = self.hybrid_model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-BJGCN-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.joint_model.eval()
        self.bone_model.eval()
        self.hybrid_model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []

            score_frag_joint = []
            pred_list_joint = []

            score_frag_bone = []
            pred_list_bone = []

            score_frag_hybrid = []
            pred_list_hybrid = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    joint_data = data[:, :3, ...]
                    bone_data = data[:, 3:, ...]
                    
                    jt_output = self.joint_model(joint_data)['logits']
                    bt_output = self.bone_model(bone_data)['logits']
                    mm_output = self.hybrid_model(data)['logits']
                    output = jt_output + bt_output + mm_output
                    
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    # acc = torch.mean((predict_label == label.data).float())
                    pred_list.append(predict_label.data.cpu().numpy())

                    score_frag_joint.append(jt_output.data.cpu().numpy())
                    _, predict_label = torch.max(jt_output.data, 1)
                    pred_list_joint.append(predict_label.data.cpu().numpy())

                    score_frag_bone.append(bt_output.data.cpu().numpy())
                    _, predict_label = torch.max(bt_output.data, 1)
                    pred_list_bone.append(predict_label.data.cpu().numpy())

                    score_frag_hybrid.append(mm_output.data.cpu().numpy())
                    _, predict_label = torch.max(mm_output.data, 1)
                    pred_list_hybrid.append(predict_label.data.cpu().numpy())

                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            score_joint = np.concatenate(score_frag_joint)
            score_bone = np.concatenate(score_frag_bone)
            score_hybrid = np.concatenate(score_frag_hybrid)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            accuracy_joint = self.data_loader[ln].dataset.top_k(score_joint, 1)
            accuracy_bone = self.data_loader[ln].dataset.top_k(score_bone, 1)
            accuracy_hybrid = self.data_loader[ln].dataset.top_k(score_hybrid, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            if accuracy_joint > self.best_joint_acc:
                self.best_joint_acc = accuracy_joint
                self.best_joint_acc_epoch = epoch + 1
            
            if accuracy_bone > self.best_bone_acc:
                self.best_bone_acc = accuracy_bone
                self.best_bone_acc_epoch = epoch + 1

            if accuracy_hybrid > self.best_hybrid_acc:
                self.best_hybrid_acc = accuracy_hybrid
                self.best_hybrid_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, 'Joint Model Accuracy: ', accuracy_joint, 'Bone Model Accuracy: ', accuracy_bone, 'Hybrid Model Accuracy: ', accuracy_hybrid, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\t Top{}: Mean {:.2f}%, Joint Model: {:.2f}%, Bone Model: {:.2f}%, Hybrid Model: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k), 100 * self.data_loader[ln].dataset.top_k(score_joint, k), 
                    100 * self.data_loader[ln].dataset.top_k(score_bone, k), 100 * self.data_loader[ln].dataset.top_k(score_hybrid, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters for one input: {count_parameters(self.joint_model)}')
            self.print_log(f'# Parameters for hybrid input: {count_parameters(self.hybrid_model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch
                self.train(epoch, save_model=save_model)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-JGCN-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.joint_model.load_state_dict(weights)

            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-BGCN-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.bone_model.load_state_dict(weights)

            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-HybridGCN-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.hybrid_model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params_single_input = sum(p.numel() for p in self.joint_model.parameters() if p.requires_grad)
            num_params_hybrid_input = sum(p.numel() for p in self.hybrid_model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}, {self.best_joint_acc}, {self.best_bone_acc}, {self.best_hybrid_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}, {self.best_joint_acc_epoch}, {self.best_bone_acc_epoch}, {self.best_hybrid_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params_single_input}, {num_params_hybrid_input}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
