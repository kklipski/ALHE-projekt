from src.ddpg_evo.evo import EvolutionaryDDPG

def main():
    evo_network = EvolutionaryDDPG(n_networks = 3, max_buffer = 100000,
                                   max_iterations= 2, max_steps = 200, episodes_ready = [2,3,4])

    evo_network.train()

if __name__ == "__main__":
    main()