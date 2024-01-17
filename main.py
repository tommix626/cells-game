# understand everything and run the program
from config import Config
from env import Env

if __name__ == '__main__':
    config = Config()
    env = Env(config)
    for i in range(config.num_generations*10):
        print("Generation: " + str(i))
        env.simulate()
        # env.print_scores()