import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np

# Save the training script and all the arguments to a file so that you
# don't feel like an idiot later when you can't replicate results
import shutil
def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def performance_stats(policies, rewards, matches):

    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    accuracy = torch.cat(matches, 0).mean()

    reward = rewards.mean()
    sparsity = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return accuracy, reward, sparsity, variance, policy_set

class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print '# setting learning_rate to %.2E'%lr


# load model weights trained using scripts from https://github.com/felixgwu/img_classification_pk_pytorch OR
# from torchvision models into our flattened resnets
def load_weights_to_flatresnet(source_model, target_model):

    # compatibility for nn.Modules + checkpoints
    if hasattr(source_model, 'state_dict'):
        source_model = {'state_dict': source_model.state_dict()}
    source_state = source_model['state_dict']
    target_state = target_model.state_dict()

    # remove the module. prefix if it exists (thanks nn.DataParallel)
    if source_state.keys()[0].startswith('module.'):
        source_state = {k[7:]:v for k,v in source_state.items()}


    common = set(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var','fc.weight', 'fc.bias'])
    for key in source_state.keys():

        if key in common:
            target_state[key] = source_state[key]
            continue

        if 'downsample' in key:
            layer, num, item = re.match('layer(\d+).*\.(\d+)\.(.*)', key).groups()
            translated = 'ds.%s.%s.%s'%(int(layer)-1, num, item)
        else:
            layer, item = re.match('layer(\d+)\.(.*)', key).groups()
            translated = 'blocks.%s.%s'%(int(layer)-1, item)


        if translated in target_state.keys():
            target_state[translated] = source_state[key]
        else:
            print translated, 'block missing'

    target_model.load_state_dict(target_state)
    return target_model






