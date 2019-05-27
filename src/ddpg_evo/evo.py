from src.ddpg.train import Trainer
from src.ddpg.buffer import MemoryBuffer
import gym
import numpy as np
import random


class EvolutionaryDDPG:
    def __init__(self, n_networks, max_buffer, max_iterations, max_steps, episodes_ready):
        self.n = n_networks                 # liczba sieci
        self.max_buffer = max_buffer
        self.max_iterations = max_iterations
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


    def run_welchs_ttest(self,idx1, idx2):
        """
        Porównanie nagród cząstkowych dwóch sieci przy użyciu Welch's t-test

        :return: indeks najlepszej sieci
        """
        return  self.n-1 # :)

    def exploit(self, idx):
        """
        Eksploatacja polega na jednolitym próbkowaniu innego (losowo wybranego) agenta w populacji,
        a następnie porównaniu ostatnich 10 cząstkowych nagród przy użyciu Welch’s t-test.
        Jeśli próbkowany agent ma wyższą średnią cząstkową nagrodę i spełnia warunki t-test,
        wagi z hiperparametrami są kopiowane do obecnego agenta.

        :param idx: indeks sieci dla której wywołujemy exploit
        """

        # losujemy indeks sieci innej od obecnej
        random_idx = random.randrange(self.n)
        while random_idx == idx:
            random_idx = random.randrange(self.n)

        # wybieramy lepszą sieć
        best_net_idx = self.run_welchs_ttest(idx, random_idx)

        # jeśli wylosowana sieć okazała się być lepsza
        if idx != best_net_idx:

            # podmieniamy wagi
            self.ddpgs[idx].set_weigths(self.ddpgs[best_net_idx].get_weigths())
            print("<exploit",idx,"> Wczytano nowe wagi z sieci nr. ",best_net_idx)
        print("<exploit",idx,"> Wagi zostają, są lepsze od sieci nr.",best_net_idx)



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

                self.append_score(ddpg_idx, reward)
                print('NETWORK ',ddpg_idx,' EPISODE : ', iteration, ' SCORE : ', reward)

                # każda sieć ma swój max epizodów, po których zostaną wywołane metody exploit i explore
                if iteration % self.episodes_ready[ddpg_idx] == 0:
                    self.exploit(ddpg_idx)
                    self.explore()

            if iteration % 100 == 0:
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
