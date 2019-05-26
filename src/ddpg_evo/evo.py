from src.ddpg.train import Trainer
from src.ddpg.buffer import MemoryBuffer
import gym
import numpy as np


class EvolutionaryDDPG:
    def __init__(self, n_networks, max_buffer, max_iterations, max_steps, episodes_ready):
        self.n = n_networks
        self.max_buffer = max_buffer
        self.max_iterations = max_iterations
        self.max_steps = max_steps
        self.episodes_ready = episodes_ready
        self.rams = []

        self.envs = self.create_envs()
        self.ddpgs = self.create_ddpg()

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
            self.rams.append(ram)
            trainer = Trainer(S_DIM, A_DIM, A_MAX, ram)

            ddpgs.append(trainer)
        return ddpgs

    def train(self):
        reward = None
        # Liczba iteracji algorytmu
        for iteration in range(self.max_iterations):

            # Dla każdej sieci
            for ddpg_idx in range(self.n):
                trainer = self.ddpgs[ddpg_idx]
                ram = self.rams[ddpg_idx]
                env = self.envs[ddpg_idx]

                # Zresetuj środowisko
                observation = env.reset()

                # Wykonaj max_steps kroków
                for r in range(self.max_steps):
                    env.render()
                    state = np.float32(observation)

                    action = trainer.get_exploration_action(state)
                    new_observation, reward, done, info = env.step(action)

                    if done:
                        new_state = None
                    else:
                        new_state = np.float32(new_observation)

                        ram.add(state, action, reward, new_state)

                    observation = new_observation

                    trainer.optimize()
                    if done:
                        break

                print('NETWORK ',ddpg_idx,' EPISODE : ', iteration, ' SCORE : ', reward)

                # każda sieć ma swój max epizodów, po których zostaną wywołane metody exploit i explore
                if iteration % self.episodes_ready[ddpg_idx] == 0:
                    self.exploit()
                    self.explore()

            if iteration % 100 == 0:
                self.save_ckpt()

    def load_ckpt(self, episode):
        for ddpg in self.ddpgs:
            ddpg.trainer.load_models_path('./Models/' + str(self.ddpgs.index(ddpg)) + '_' + str(episode) + '_actor.pt',
                                          './Models/' + str(self.ddpgs.index(ddpg)) + '_' + str(episode) + '_critic.pt')

    def save_ckpt(self):
        for ddpg in self.ddpgs:
            idx_ddpg = self.ddpgs.index(ddpg)
            ddpg.trainer.save_models_path(idx_ddpg, self.episodes_counter[idx_ddpg])
