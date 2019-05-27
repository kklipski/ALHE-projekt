from src.ddpg.train import Trainer
from src.ddpg.buffer import MemoryBuffer
import gym
import numpy as np
import random
import scipy.stats
import math
from statistics import mean
import torch

class EvolutionaryDDPG:
    def __init__(self, n_networks, max_buffer, max_episodes, max_steps, episodes_ready):
        self.n = n_networks                 # liczba sieci
        self.max_buffer = max_buffer
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.episodes_ready = episodes_ready
        self.rams = []

        # początkowe ostatnie 10 cząstkowych wyników dla wszystkich sieci ustawiamy na -100
        self.last_ten_scores = [[-100 for _ in range(10)] for _ in range(self.n)]

        self.envs = self.create_envs()
        self.ddpgs = self.create_ddpg()


    def append_score(self, idx, new_score):
        """
        Usuwa ostatni wynik z 10 ostatnich cząstkowych wyników i dodaje nowy
        :param idx: indeks sieci
        :param new_score: nowy wynik
        """
        self.last_ten_scores[idx] = self.last_ten_scores[idx][1:]
        self.last_ten_scores[idx].append(new_score)


    def pick_net(self,idx1, idx2):
        """
        Porównanie nagród cząstkowych dwóch sieci przy użyciu Welch's t-test

        :param idx1: obecna sieć
        :param idx2: losowo wybrana sieć
        :return: indeks najlepszej sieci
        """

        # t, pvalue = scipy.stats.ttest_ind(self.last_ten_scores[idx1], self.last_ten_scores[idx2], equal_var=False)
        # if math.isnan(t) or math.isnan(pvalue):         # jeśli nie przeszedł welch's t-test TODO: na konsultacje
        #     return idx1
        if mean(self.last_ten_scores[idx1]) >  mean(self.last_ten_scores[idx2]):   # porównanie średnich z ostatnich 10 wyników
            return  idx1 # :)
        else:                     # przeszło welchs t-test i średnia jest większa
            return idx2

    def exploit(self, idx):
        """
        Eksploatacja polega na jednolitym próbkowaniu innego (losowo wybranego) agenta w populacji,
        a następnie porównaniu ostatnich 10 cząstkowych nagród przy użyciu Welch’s t-test.
        Jeśli próbkowany agent ma wyższą średnią cząstkową nagrodę i spełnia warunki t-test,
        wagi z hiperparametrami są kopiowane do obecnego agenta.

        :param idx: indeks sieci dla której wywołujemy exploit
        """

        # losujemy indeks sieci różnej od obecnej
        random_idx = random.randrange(self.n)
        while random_idx == idx:
            random_idx = random.randrange(self.n)

        # wybieramy lepszą sieć
        best_net_idx = self.pick_net(idx, random_idx)

        # jeśli wylosowana sieć okazała się być lepsza
        if idx != best_net_idx:

            # podmieniamy wagi
            self.ddpgs[idx].set_weights(self.ddpgs[best_net_idx].get_weights())
            print("<exploit",idx,"> Wczytano nowe wagi z sieci nr. ",best_net_idx)
        else:
            print("<exploit",idx,"> Wagi zostają, są lepsze od sieci nr.",best_net_idx)


    def explore(self, idx):
        net = self.ddpgs[idx]

        self.ddpgs[idx].critic.fca1.weight = torch.mul(self.ddpgs[idx].critic.fca1.weight, 0.7)
        self.ddpgs[idx].critic.fc2.weight =  self.ddpgs[idx].critic.fc2.weight * 0.7
        self.ddpgs[idx].critic.fc3.weight =  self.ddpgs[idx].critic.fc3.weight * 0.7
        self.ddpgs[idx].critic.fcs1.weight =  self.ddpgs[idx].critic.fcs1.weight * 0.7
        self.ddpgs[idx].critic.fcs2.weight = self.ddpgs[idx].critic.fcs2.weight * 0.7

        self.ddpgs[idx].actor.fc1.weight =  self.ddpgs[idx].critic.fc1.weight * 0.7
        self.ddpgs[idx].actor.fc2.weight = self.ddpgs[idx].critic.fc2.weight * 0.7
        self.ddpgs[idx].actor.fc3.weight = self.ddpgs[idx].critic.fc3.weight * 0.7
        self.ddpgs[idx].actor.fc4.weight = self.ddpgs[idx].critic.fc4.weight * 0.7
        # weights_critic, weights_actor = net.get_weights()
        #
        # for idx, _ in enumerate(weights_critic):
        #     weights_critic[idx] = weights_critic[idx]*0.7
        #
        # for idx, _ in enumerate(weights_actor):
        #     weights_critic[idx] = weights_critic[idx]*0.7
        #
        # net.set_weights((weights_critic,weights_actor))


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
        for episode in range(self.max_episodes):

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

                self.append_score(ddpg_idx, reward)
                print('NETWORK ',ddpg_idx,' EPISODE : ', episode, ' SCORE : ', reward)

                # każda sieć ma swój max epizodów, po których zostaną wywołane metody exploit i explore
                if episode % self.episodes_ready[ddpg_idx] == 0:
                    self.exploit(ddpg_idx)
                    self.explore(ddpg_idx)

            if episode % 100 == 0:
                # self.save_ckpt(iteration)
                pass
    def load_ckpt(self, episode):
        for ddpg in self.ddpgs:
            ddpg.load_models_path('./Models/' + str(self.ddpgs.index(ddpg)) + '_' + str(episode) + '_actor.pt',
                                          './Models/' + str(self.ddpgs.index(ddpg)) + '_' + str(episode) + '_critic.pt')

    def save_ckpt(self, iteration):
        for ddpg in self.ddpgs:
            idx_ddpg = self.ddpgs.index(ddpg)
            ddpg.save_models_path(idx_ddpg, iteration)
