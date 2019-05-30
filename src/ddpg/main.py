from __future__ import division

import gc
import gym
import numpy as np

from src.ddpg.buffer import MemoryBuffer
from src.ddpg.train import Trainer


def train_ddpg(actor_path=None, critic_path=None):
    env = gym.make('BipedalWalker-v2')

    max_episodes = 5000
    max_steps = 1000
    max_buffer = 1000000
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_max = env.action_space.high[0]

    print(' State Dimensions :- ', s_dim)
    print(' Action Dimensions :- ', a_dim)
    print(' Action Max :- ', a_max)

    ram = MemoryBuffer(max_buffer)
    trainer = Trainer(s_dim, a_dim, a_max, ram)
    # trainer.load_models_path(r"C:\Users\rzaro\Repositories\ALHE-projekt\SavedModels\Models\100_actor.pt",
    #                          r"C:\Users\rzaro\Repositories\ALHE-projekt\SavedModels\Models\100_critic.pt")
    for _ep in range(max_episodes):
        observation = env.reset()
        print('EPISODE :- ', _ep)
        for r in range(max_steps):
            # env.render()
            state = np.float32(observation)

            action = trainer.get_exploration_action(state)
            new_observation, reward, done, info = env.step(action)

            if not done:
                new_state = np.float32(new_observation)
                # push this exp in ram
                ram.add(state, action, reward, new_state)

            observation = new_observation

            # perform optimization
            trainer.optimize()
            if done:
                break

        # check memory consumption and clear memory
        gc.collect()
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)

        if _ep % 100 == 0:
            trainer.save_models(_ep)

    print('Completed episodes')


def main():
    train_ddpg()


if __name__ == "__main__":
    main()
