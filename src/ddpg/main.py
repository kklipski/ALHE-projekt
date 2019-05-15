from __future__ import division

import gc
import gym
import numpy as np

from src.ddpg.buffer import MemoryBuffer
from src.ddpg.train import Trainer

def main():
    env = gym.make('BipedalWalker-v2')

    MAX_EPISODES = 5000
    MAX_STEPS = 1000
    MAX_BUFFER = 1000000
    MAX_TOTAL_REWARD = 300
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]

    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)

    ram = MemoryBuffer(MAX_BUFFER)
    trainer = Trainer(S_DIM, A_DIM, A_MAX, ram)

    for _ep in range(MAX_EPISODES):
        observation = env.reset()
        print('EPISODE :- ', _ep)
        for r in range(MAX_STEPS):
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

        # check memory consumption and clear memory
        gc.collect()
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)

        if _ep % 100 == 0:
            trainer.save_models(_ep)

    print('Completed episodes')

if __name__ == "__main__":
    main()
