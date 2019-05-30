from src.ddpg.train import Trainer
from src.ddpg.buffer import MemoryBuffer
from statistics import mean

import gym
import numpy as np
import random
import scipy.stats


class EvolutionaryDDPG:
    def __init__(self, n_networks, max_buffer, max_episodes, max_steps, episodes_ready, explore_prob, explore_factors):
        self.n = n_networks  # liczba sieci
        self.max_buffer = max_buffer
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.episodes_ready = episodes_ready
        if len(self.episodes_ready) < n_networks:
            print("episodes_ready.len() != n_networks")
            raise Exception

        self.explore_prob = explore_prob - int(explore_prob)
        self.explore_factors = explore_factors

        self.rams = []

        # początkowe ostatnie 10 cząstkowych wyników dla wszystkich sieci ustawiamy na -100
        self.last_ten_scores = [[-100 for _ in range(10)] for _ in range(self.n)]

        self.envs = self.create_envs()
        self.ddpgs = self.create_ddpg()

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
            s_dim = env.observation_space.shape[0]
            a_dim = env.action_space.shape[0]
            a_max = env.action_space.high[0]

            print(' State Dimensions :- ', s_dim)
            print(' Action Dimensions :- ', a_dim)
            print(' Action Max :- ', a_max)

            ram = MemoryBuffer(self.max_buffer)
            self.rams.append(ram)
            trainer = Trainer(s_dim, a_dim, a_max, ram)

            ddpgs.append(trainer)
        return ddpgs

    def exploit(self, idx):
        """
        Eksploatacja polega na jednolitym próbkowaniu innego (losowo wybranego) agenta w populacji,
        a następnie porównaniu ostatnich 10 cząstkowych nagród przy użyciu Welch’s t-test.
        Jeśli próbkowany agent ma wyższą średnią cząstkową nagrodę i spełnia warunki t-test,
        wagi z hiperparametrami są kopiowane do obecnego agenta.

        :param idx: indeks sieci, dla której wywołujemy exploit()
        """

        # losujemy indeks sieci różnej od obecnej
        random_idx = random.randrange(self.n)
        while random_idx == idx:
            random_idx = random.randrange(self.n)

        # wybieramy lepszą sieć
        best_net_idx = self.pick_net(idx, random_idx)

        # jeśli wylosowana sieć okazała się być lepsza
        if idx != best_net_idx:

            # OLD METHOD
            # podmieniamy wagi
            # self.ddpgs[idx].actor.load_state_dict(self.ddpgs[best_net_idx].actor.state_dict())
            # self.ddpgs[idx].critic.load_state_dict(self.ddpgs[best_net_idx].critic.state_dict())
            # podobno potrzebne, jeśli chcemy kontynuować trenowanie, albo eval() zamiast train()
            # self.ddpgs[idx].actor.train()
            # self.ddpgs[idx].critic.train()

            # podmieniamy wagi
            new_param = self.ddpgs[best_net_idx].actor.parameters()
            for param in self.ddpgs[idx].actor.parameters():
                param.data.copy_(next(new_param))
            new_param = self.ddpgs[best_net_idx].critic.parameters()
            for param in self.ddpgs[idx].critic.parameters():
                param.data.copy_(next(new_param))

            print("<exploit", idx, "> Wczytano nowe wagi z sieci nr ", best_net_idx)
        else:
            print("<exploit", idx, "> Wagi zostają, są lepsze od sieci nr ", random_idx)

    def explore(self, idx):
        # VERY OLD METHOD
        # net = self.ddpgs[idx]

        if random.random() < 0.5:
            # OLD METHOD
            # self.ddpgs[idx].multiply_critic(self.explore_factors[0])
            # self.ddpgs[idx].multiply_actor(self.explore_factors[0])

            for param in self.ddpgs[idx].actor.parameters():
                param.data.mul_(self.explore_factors[0])
            for param in self.ddpgs[idx].critic.parameters():
                param.data.mul_(self.explore_factors[0])
            print("<explore", idx, "> Przemnożono wagi przez ", self.explore_factors[0])
        else:
            # OLD METHOD
            # self.ddpgs[idx].multiply_critic(self.explore_factors[1])
            # self.ddpgs[idx].multiply_actor(self.explore_factors[1])

            for param in self.ddpgs[idx].actor.parameters():
                param.data.mul_(self.explore_factors[1])
            for param in self.ddpgs[idx].critic.parameters():
                param.data.mul_(self.explore_factors[1])
            print("<explore", idx, "> Przemnożono wagi przez ", self.explore_factors[1])

        # VERY OLD METHOD
        # weights_critic, weights_actor = net.get_weights()
        #
        # for idx, _ in enumerate(weights_critic):
        #     weights_critic[idx] = weights_critic[idx]*0.7
        #
        # for idx, _ in enumerate(weights_actor):
        #     weights_critic[idx] = weights_critic[idx]*0.7
        #
        # net.set_weights((weights_critic,weights_actor))

    def pick_net(self, idx1, idx2):
        """
        Porównanie nagród cząstkowych dwóch sieci przy użyciu Welch's t-test
        :param idx1: obecna sieć
        :param idx2: losowo wybrana sieć
        :return: indeks najlepszej sieci
        """

        statistic, pvalue = scipy.stats.ttest_ind(self.last_ten_scores[idx1], self.last_ten_scores[idx2],
                                                  equal_var=False)
        if pvalue <= 0.05:
            # przeszło welch's t-test, teraz porównanie średnich z ostatnich 10 wyników
            if mean(self.last_ten_scores[idx1]) > mean(self.last_ten_scores[idx2]):
                return idx1  # obecna sieć lepsza
            else:
                return idx2  # losowo wybrana sieć lepsza
        else:
            return idx1  # nie przeszło welch's t-test

    def train(self):
        # Liczba iteracji algorytmu
        for episode in range(self.max_episodes):

            # Dla każdej sieci
            for ddpg_idx in range(self.n):
                trainer = self.ddpgs[ddpg_idx]
                ram = self.rams[ddpg_idx]
                env = self.envs[ddpg_idx]

                # Zresetuj środowisko
                observation = env.reset()

                # Zliczamy całkowity uzyskany wynik
                total_reward = 0

                # Wykonaj max_steps kroków
                for r in range(self.max_steps):
                    # env.render()
                    state = np.float32(observation)

                    action = trainer.get_exploration_action(state)
                    new_observation, reward, done, info = env.step(action)
                    total_reward = total_reward + reward

                    if not done:
                        new_state = np.float32(new_observation)
                        ram.add(state, action, reward, new_state)

                    observation = new_observation

                    trainer.optimize()
                    if done:
                        break

                self.append_score(ddpg_idx, total_reward)
                print('NETWORK ', ddpg_idx, ' EPISODE : ', episode, ' SCORE : ', total_reward)

                # każda sieć ma swój max epizodów, po których zostaną wywołane metody exploit i explore
                if episode % self.episodes_ready[ddpg_idx] == 0 and episode != 0:
                    self.exploit(ddpg_idx)
                    if random.random() < self.explore_prob:
                        self.explore(ddpg_idx)

            if episode % 100 == 0:
                self.save_ckpt(episode)

    def append_score(self, idx, new_score):
        """
        Usuwa ostatni wynik z 10 ostatnich cząstkowych wyników i dodaje nowy
        :param idx: indeks sieci
        :param new_score: nowy wynik
        """
        self.last_ten_scores[idx] = self.last_ten_scores[idx][1:]
        self.last_ten_scores[idx].append(new_score)

    def save_ckpt(self, episode):
        idx_ddpg = 0
        for ddpg in self.ddpgs:
            ddpg.save_models_path(idx_ddpg, episode)
            idx_ddpg = idx_ddpg + 1
        print('Models saved successfully')

    def load_ckpt(self, episode):
        idx_ddpg = 0
        for ddpg in self.ddpgs:
            ddpg.load_models_path('Models/' + str(idx_ddpg) + '_' + str(episode) + '_actor.pt',
                                  'Models/' + str(idx_ddpg) + '_' + str(episode) + '_critic.pt')
            idx_ddpg = idx_ddpg + 1
        print('Models loaded successfully')
