import os
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import tqdm
import utils
import torch.optim as optim
from torch.distributions import Bernoulli

from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
from torch.utils.data import Dataset, DataLoader

from models import resnet, base

from scipy.spatial.distance import cdist

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--data_dir', default='dataset-path', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load rnet+agent from')
parser.add_argument('--pretrained', default=None, help='pretrained policy model checkpoint (from curriculum training)')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epoch_step', type=int, default=30, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=200, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
# parser.add_argument('--joint', action ='store_true', default=True, help='train both the policy network and the resnet')
parser.add_argument('--penalty', type=float, default=-5, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def get_reward(preds, targets, policy):
    diversity_reward = policy.sum(1)
    dist = cdist(policy, policy, metric='hamming')
    d = dist.mean(1)
    for i in range(policy.size()[0]):
        diversity_reward[i] = d[i]

    # diversity term
    diversity_reward = 1 - (1 - diversity_reward.float()) ** 2

    # sparsity term
    block_use = policy.sum(1).float() / policy.size(1)
    sparse_reward = 1.0 - block_use ** 2

    _, pred_idx = preds.max(1)
    match = (pred_idx == targets).data
    
    # reward function
    reward = diversity_reward * 0.6 + sparse_reward * 0.4
    reward[1 - match] = args.penalty
    reward = reward.unsqueeze(1)

    return reward, match.float()


def train(epoch):

    agent.train()
    rnet.train()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, locations, targets) in enumerate(trainloader):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        locations = Variable(locations)

        inputs = inputs.cuda()
        locations = locations.cuda()

        probs = agent(inputs, locations)

        #---------------------------------------------------------------------#

        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        probs = probs*args.alpha + (1-probs)*(1-args.alpha)
        distr = Bernoulli(probs)
        policy = distr.sample()

        v_inputs = Variable(inputs.data, volatile=True)
        preds_map = rnet.forward(v_inputs, policy_map)
        preds_sample = rnet.forward(inputs, policy)

        reward_map, _ = get_reward(preds_map, targets, policy_map.data)
        reward_sample, match = get_reward(preds_sample, targets, policy.data)

        advantage = reward_sample - reward_map

        loss = -distr.log_prob(policy).sum(1, keepdim=True) * Variable(advantage)
        loss = loss.sum()

        #---------------------------------------------------------------------#
        loss += F.cross_entropy(preds_sample, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        matches.append(match.cpu())
        rewards.append(reward_sample.cpu())
        policies.append(policy.data.cpu())

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    log_str = 'E: %d | A: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d'%(epoch, accuracy, reward, sparsity, variance, len(policy_set))
    print log_str


def test(epoch):

    agent.eval()
    rnet.eval()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, locations, targets) in enumerate(testloader):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets).cuda(async=True)
        locations = Variable(locations, volatile=True)

        inputs = inputs.cuda()
        locations = locations.cuda()

        probs = agent(inputs, locations)

        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0
        policy = Variable(policy)

        preds = rnet.forward(inputs, policy)
        reward, match = get_reward(preds, targets, policy.data)

        matches.append(match)
        rewards.append(reward)
        policies.append(policy.data)

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    log_str = 'TS - A: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d'%(accuracy, reward, sparsity, variance, len(policy_set))
    print log_str

    # save the model
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()

    state = {
      'agent': agent_state_dict,
      'resnet': rnet_state_dict,
      'epoch': epoch,
      'reward': reward,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_R_%.2E_S_%.2f_#_%d.t7'%(epoch, accuracy, reward, sparsity, len(policy_set)))


#--------------------------------------------------------------------------------------------------------#
# dataset path
root = args.data_dir

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        locations = []
        labels = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('\t')

            imgs.append(words[0])
            locations.append(torch.Tensor(list(map(float, eval(words[1])))))
            labels.append(int(words[2]))

        self.imgs = imgs
        self.locations = locations
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)

        location = self.locations[index]
        label = self.labels[index]
        #if self.target_transform is not None:
            #location = self.target_transform(location)
            #label = self.target_transform(label)

        return img, location, label

    def __len__(self):
        return len(self.imgs)


# Data loading code
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform_train = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainset = MyDataset(txt=root + 'Landmark-732_train.txt', transform=transform_train, target_transform=transforms.ToTensor())
testset = MyDataset(txt=root + 'Landmark-732_val.txt', transform=transform_test, target_transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# create model
layer_config = [7, 7, 7]
rnet = resnet.FlatResNet224(base.Bottleneck, layer_config, num_classes=732)     #recognition network
agent = resnet.Policy224_loc([1,1,1,1], num_blocks=21)    #policy network

agent = nn.DataParallel(agent)
rnet = nn.DataParallel(rnet)
rnet.cuda()
agent.cuda()

#load pre-trained recognition network
rnet_checkpoint = torch.load('pre-trained recognition network model path')
rnet.load_state_dict(rnet_checkpoint['state_dict'])

#load trained policy network in the pre-training phase
load_agent = 'trained policy network model path'
checkpoint = torch.load(load_agent)
agent.load_state_dict(checkpoint['agent'])
print 'loaded pretrained model from', load_agent

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    rnet.load_state_dict(checkpoint['resnet'])
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print 'loaded agent from', args.load

optimizer = optim.Adam(list(agent.parameters())+list(rnet.parameters()), lr=args.lr, weight_decay=args.wd)

#configure(args.cv_dir+'/log', flush_secs=5)
lr_scheduler = utils.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)
    train(epoch)
    if epoch%1==0:
        test(epoch)
