from src.ddpg.train import Trainer
from src.ddpg.buffer import MemoryBuffer
import gym


class EvolutionaryDDPG:
    def __init__(self, n_networks, max_buffer):
        self.n = n_networks
        self.ddpgs = self.create_ddpg()
        self.envs = self.create_envs()
        self.max_buffer = max_buffer

    def exploit(self):
        pass

    def explore(self):
        pass

    def create_envs(self):
        envs = []
        for i in range(self.n):
            env = gym.make('BipedalWalker-v2')
            envs.append(env)

        return envs

    def create_ddpg(self):
        ddpgs = []
        for i in range(self.n):
            env = self.envs[i]
            S_DIM = env.observation_space.shape[0]
            A_DIM = env.action_space.shape[0]
            A_MAX = env.action_space.high[0]

            print(' State Dimensions :- ', S_DIM)
            print(' Action Dimensions :- ', A_DIM)
            print(' Action Max :- ', A_MAX)

            ram = MemoryBuffer(self.max_buffer)
            trainer = Trainer(S_DIM, A_DIM, A_MAX, ram)

            ddpgs.append(trainer)
        return ddpgs

    def train(self):
        pass

    def load_ckpt(self):
        pass

    def save_ckpt(self):
        for ddpg in self.ddpgs:
            ddpg.trainer.save_model(self.ddpgs.index(ddpg))    # TODO: add episode_count as argument

