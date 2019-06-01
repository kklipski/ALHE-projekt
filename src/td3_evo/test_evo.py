from src.td3_evo.evo import EvolutionaryTD3


def main():
    evo_network = EvolutionaryTD3(n_networks=4, max_buffer=100000, max_episodes=9999999, max_steps=700,
                                  episodes_ready=[5, 10, 20, 30], explore_prob=0.05, explore_factors=[0.98, 1.02])
    evo_network.train()


if __name__ == "__main__":
    main()