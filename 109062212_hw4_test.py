from model import Actor, Critic
# from wrappers import *
import torch
from utils import as_list
import numpy as np
# from osim.env import L2M2019Env

class Agent:
    def __init__(self):
        self.state_dim = 339
        self.action_dim = 22
        self.action_range = [np.zeros(self.action_dim), np.ones(self.action_dim)]
        self.skip = 4
        self.device = torch.device("cpu")
        self.actor = Actor(self.state_dim, self.action_dim, 
                            self.action_range).to(self.device)
        # self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        # self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.skip_count = 0
        self.prev_action = None

        print("action range: ", self.action_range[0], self.action_range[1])

        self.load()
    
    def load(self, filename="109062212_hw4_data"):
        ckpt = torch.load(filename)
        self.actor.load_state_dict(ckpt["actor"])
        # self.critic.load_state_dict(ckpt["critic"])
        # self.critic_target.load_state_dict(ckpt["critic_target"])
        # self.log_alpha = ckpt["log_alpha"]

    def act(self, observation):
        if self.skip_count % self.skip == 0:
            if isinstance(observation, dict):
                observation = as_list(observation)
            state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
            action = self.actor.mean(state)
            # action, _ = self.actor.sample(state)
            self.prev_action = action.cpu().detach().numpy().flatten()
        self.skip_count += 1
        return self.prev_action
'''
if __name__ == "__main__":
    env = L2M2019Env(visualize=False, difficulty=2)
    print("action range: ", env.action_space.low, env.action_space.high)
    print("obs space: ", env.observation_space.shape)
    agent = Agent()

    rewards = []
    for i in range(2):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            action = agent.act(state)
            try:
                next_state, reward, done, _ = env.step(action)
            except Exception as e:
                print(e, "fuck")
                done = True
            # env.render()
            total_reward += reward
            state = next_state
            if step % 20 == 0:
                print(f"\rStep: {step:03}, Reward: {reward:.4f},"
                    f" Total reward: {total_reward:.4f}", end="")
            # print(", ".join([f"{i:.3f}" for i in action]))
            step += 1

        print(f"Total reward: {total_reward}")
        rewards.append(total_reward)
    print(f"Average reward: {sum(rewards) / len(rewards):.4f}")
'''