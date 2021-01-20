from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_pytorch.ppo import RolloutStorage
import horovod.torch as hvd


class PPOHorovod:
    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 device='cpu',
                 log_dir='run',
                 is_testing=False):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space

        self.device = device

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.action_space.shape, init_noise_std)
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
                                      self.action_space.shape, self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # Add Horovod Distributed Optimizer
        self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.actor_critic.named_parameters())

        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(self.actor_critic.state_dict(), root_rank=0)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = log_dir

        if hvd.local_rank() == 0:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()  # .astype(np.float32)
        #current_obs = torch.from_numpy(obs).to(self.device)

        if self.is_testing:
            while True:
                with torch.no_grad():
                    current_obs = self.vec_env.reset()  # .astype(np.float32)
                    #current_obs = torch.from_numpy(obs).to(self.device)

                    # Compute the action
                    actions, actions_log_prob, values = self.actor_critic.act(current_obs)
                    #clipped_actions = np.clip(actions.cpu().numpy(), self.vec_env.action_space.low, self.vec_env.action_space.high)
                    #clipped_actions = torch.tensor(clipped_actions, dtype=torch.float).cuda()
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    # Record the transition
                    #next_obs = torch.from_numpy(next_obs).to(self.device)
                    #rews = torch.from_numpy(rews).to(self.device)
                    #dones = torch.from_numpy(dones).to(self.device)
                    #self.storage.add_transitions(current_obs, actions, rews, dones, values, actions_log_prob)
                    # current_obs.copy_(next_obs)
                    # Book keeping
                    # for info in infos:
                    #ep_info = info.get('episode')
                    # if ep_info is not None:
                    # ep_infos.append(ep_info)
        else:
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    current_obs = self.vec_env.reset()  # .astype(np.float32)
                    #current_obs = torch.from_numpy(obs).to(self.device)

                    # Compute the action
                    actions, actions_log_prob, values = self.actor_critic.act(current_obs)
                    # TODO: add action regularization
                    #clipped_actions = np.clip(actions.cpu().numpy(), self.vec_env.action_space.low, self.vec_env.action_space.high)
                    #clipped_actions = torch.tensor(clipped_actions, dtype=torch.float).cuda()
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    # Record the transition
                    # next_obs = torch.from_numpy(next_obs).to(self.device)
                    # rews = torch.from_numpy(rews).to(self.device)
                    # dones = torch.from_numpy(dones).to(self.device)
                    self.storage.add_transitions(current_obs, actions, rews, dones, values, actions_log_prob)
                    current_obs.copy_(next_obs)
                    # Book keeping
                    for info in infos:
                        ep_info = info.get('episode')
                        if ep_info is not None:
                            ep_infos.append(ep_info)

                    cur_reward_sum[:] += rews
                    cur_episode_length[:] += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                # reward_sum = [x[0] for x in reward_sum]
                # episode_length = [x[0] for x in episode_length]
                rewbuffer.extend(reward_sum)
                lenbuffer.extend(episode_length)

                _, _, last_values = self.actor_critic.act(current_obs)
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start

                mean_value_loss_avg = hvd.allreduce(torch.tensor(mean_value_loss)).item()
                mean_surrogate_loss_avg = hvd.allreduce(torch.tensor(mean_surrogate_loss)).item()
                rew_avg = hvd.allreduce(torch.tensor(statistics.mean(rewbuffer))).item()
                len_avg = hvd.allreduce(torch.tensor(statistics.mean(lenbuffer))).item()

                if hvd.local_rank() == 0:
                    self.log(locals())
                    if it % log_interval == 0:
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()

            if hvd.local_rank() == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=31):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                value = np.mean([ep_info[key] for ep_info in locs['ep_infos']])
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss_avg'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss_avg'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Train/mean_reward', locs['rew_avg'], locs['it'])
        self.writer.add_scalar('Train/mean_episode_length', locs['len_avg'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "
        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                          'collection_time']:.2f}s, learning {locs['learn_time']:.2f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {locs['mean_value_loss_avg']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss_avg']:.4f}\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                      f"""{'Mean reward:':>{pad}} {locs['rew_avg']:.4f}\n"""
                      f"""{'Mean episode length:':>{pad}} {locs['len_avg']:.4f}\n""")
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.1f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.1f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]

                actions_log_prob_batch, entropy_batch, value_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                actions_batch)
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss
