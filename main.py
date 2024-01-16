# understand everything and run the program
from config import Config
from env import Env

if __name__ == '__main__':
    config = Config()
    env = Env(config)
    for i in range(config.num_generations):
        print("Generation: " + str(i))
        for j in range(10):
            env.simulate()
        env.print_rankings()