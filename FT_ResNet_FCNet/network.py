import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from PIL import ImageDraw
from torchvision.datasets.folder import *
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class MyImageFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = default_loader,
            transform: Optional[Callable] = None,
            transform_crop: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        
        self.transform_crop = transform_crop
        self.image_paths = [sample[0] for sample in self.samples]

    def __getitem__(self, index: int):

        path, targets = self.samples[index]
        inputs = self.loader(path)
        width, height = inputs.size
        if width >= height:
            resize_height = int(96 * (height/width))
            up_padding = int(48 * (1-height/width))
            down_padding = (96 - resize_height) - up_padding
            inputs = inputs.resize((96, resize_height))
            inputs = transforms.Pad([0, up_padding, 0, down_padding], fill=0, padding_mode='constant')(inputs)
        if width < height:
            resize_width = int(96 * (width/height))
            left_padding = int(48 * (1-width/height))
            right_padding = (96 - resize_width) - left_padding
            inputs = inputs.resize((resize_width, 96))
            inputs = transforms.Pad([left_padding, 0, right_padding, 0], fill=0, padding_mode='constant')(inputs)
        
        if self.transform_crop is not None:
            inputs = self.transform_crop(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return inputs, targets, index
    
    def get_patch(self, index: int, action):

        patches = []
        length = []
        coordinate_x = []
        coordinate_y = []

        for i, idx in enumerate(index):
            path, target = self.samples[idx]
            sample_up = self.loader(path)
            width, height = sample_up.size

            if width >= height:
                up_padding = int((width - height) / 2)
                down_padding = width - height - up_padding
                sample_up = transforms.Pad([0, up_padding, 0, down_padding], fill=0, padding_mode='constant')(sample_up)
            if width < height:
                left_padding = int((height - width) / 2)
                right_padding = height - width - left_padding
                sample_up = transforms.Pad([left_padding, 0, right_padding, 0], fill=0, padding_mode='constant')(sample_up)

            width, height = sample_up.size
            sample_draw = sample_up

            for j in range(3):
                if j == 0:
                    length.append(torch.floor(action[i][j] * height))
                if j == 1:
                    coordinate_x.append(torch.floor(action[i][j] * (width - length[0].item())).int())
                if j == 2:
                    coordinate_y.append(torch.floor(action[i][j] * (height - length[0].item())).int())

            sample = sample_up.crop((coordinate_x[0].item(), coordinate_y[0].item(), coordinate_x[0].item() + length[0].item(), coordinate_y[0].item() + length[0].item()))
            sample = sample.resize((224, 224))

            length = []
            coordinate_x = []
            coordinate_y = []

            if self.transform is not None:
                sample = self.transform(sample)

            if self.target_transform is not None:
                targets = self.target_transform(targets)

            patches.append(sample)

        patch_stacked = torch.stack(patches, dim=0)

        return patch_stacked

    def get_original_image_path(self, index: int):
        return self.image_paths[index]


class model_ResNet(nn.Module):
    def __init__(self, model):

        super(model_ResNet, self).__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
    
    def forward(self, x):
        x = self.backbone(x)
        output = self.max(x)
        output = output.view(x.size(0), -1)

        return output, x.detach()
    

class model_ResNet_Crop(nn.Module):
    def __init__(self, model):

        super(model_ResNet_Crop, self).__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=3, stride=3)
    
    def forward(self, x):
        x = self.backbone(x)
        output = self.max(x)
        output = output.view(x.size(0), -1)

        return output, x.detach()
    

class Full_layer(torch.nn.Module):
    def __init__(self, feature_num, feature_size, classes_num):
        super(Full_layer, self).__init__()
        self.class_num = classes_num
        self.feature_num = feature_num
        self.feature_size = feature_size

        self.features = nn.Sequential(
            nn.BatchNorm1d(feature_num),
            nn.Linear(feature_num, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):
        output = self.features(x)
        output = self.classifier(output)

        return output
    

class Crop_layer(torch.nn.Module):
    def __init__(self, feature_num, classes_num):
        super(Crop_layer, self).__init__()
        self.class_num = classes_num
        self.feature_num = feature_num

        self.features = nn.Sequential(
            nn.BatchNorm1d(feature_num),
            nn.ELU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_num, classes_num),
        )

    def forward(self, x):
        output = self.features(x)
        output = self.classifier(output)

        return output
    

class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_state_dim, action_std=0.1):
        super(ActorCritic, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, hidden_state_dim),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, 3),
            nn.Sigmoid()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1)
        )
        
        self.action_var = torch.full((3,), action_std).cuda()
        self.hidden_state_dim = hidden_state_dim
        self.feature_dim = feature_dim
        self.feature_ration = int(math.sqrt(state_dim/feature_dim))

    def forward(self):
        raise NotImplementedError
    
    def act(self, state_ini, memory, training):

        state = state_ini.view(state_ini.size(0), -1)

        state = self.state_encoder(state)
        action_mean = self.actor(state)

        cov_mat = torch.diag(self.action_var).cuda()
        dist = torch.distributions.multivariate_normal.MultivariateNormal(action_mean, scale_tril=cov_mat)

        action = dist.sample().cuda()
        if training:
            action = F.relu(action)
            action = 1 - F.relu(1 - action)
            action_logprob = dist.log_prob(action).cuda()
            memory.states.append(state_ini)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        else:
            action = action_mean
        
        return action.detach()
    
    def evaluate(self, state, action):
        

        seq_l = state.size(0)
        batch_size = state.size(1)

        state = state.view(batch_size, -1)

        state = self.state_encoder(state)
        
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).cuda()
        dist = torch.distributions.multivariate_normal.MultivariateNormal(action_mean, scale_tril=cov_mat)

        action_logprobs = dist.log_prob(torch.squeeze(action.view(batch_size, -1))).cuda()

        dist_entropy = dist.entropy().cuda()
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)


class PPO:
    def __init__(self, feature_dim, state_dim, hidden_state_dim, action_std=0.1, 
                 lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(feature_dim, state_dim, hidden_state_dim, action_std).cuda()    
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(feature_dim, state_dim, hidden_state_dim, action_std).cuda()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory, training=True):
        return self.policy_old.act(state, memory, training)
    
    def update(self, memory):
        rewards = memory.rewards[0]

        old_states = torch.stack(memory.states, 0).cuda().detach()
        old_actions = torch.stack(memory.actions, 0).cuda().detach()
        old_logprobs = torch.stack(memory.logprobs, 0).cuda().detach()


        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = - torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())