from src.ddpg.train import Trainer
from src.ddpg.buffer import MemoryBuffer
import gym
import numpy as np


class EvolutionaryDDPG:
    def __init__(self, n_networks, max_buffer, max_episodes, max_steps, episodes_ready ):
        self.n = n_networks
        self.max_buffer = max_buffer
        self.max_episodes = max_episodes
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
        # Ile razy ma się wywołać 100 epizodów
        for step in range(self.max_episodes):

            # Dla każdej sieci
            for ddpg_idx in range(self.n):
                trainer = self.ddpgs[ddpg_idx]
                ram = self.rams[ddpg_idx]
                env = self.envs[ddpg_idx]

                # Dla 100 epizodów
                for _ep in range(self.episodes_ready):
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
                            # push this exp in ram
                            ram.add(state, action, reward, new_state)

                        observation = new_observation

                        # perform optimization
                        trainer.optimize()
                        if done:
                            break

                    print('NETWORK ',ddpg_idx,' EPISODE : ', _ep + self.episodes_ready * step, ' SCORE : ', reward)

    def load_ckpt(self):
        pass

    def save_ckpt(self):
        for ddpg in self.ddpgs:
            ddpg.trainer.save_model(self.ddpgs.index(ddpg))    # TODO: add episode_count as argument

