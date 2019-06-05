import gym
from src.td3.TD3 import TD3


def test():
    episode_number = 300
    network_index = 0
    n_episodes = 3
    lr = 0.002
    max_timesteps = 2000
    render = True

    filename = 'TD3_ep_{}_net_{}'.format(episode_number, network_index)
    directory = './preTrained'

    env = gym.make("BipedalWalker-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)

    policy.load_actor(directory, filename)

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break

        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        env.close()


if __name__ == '__main__':
    test()



