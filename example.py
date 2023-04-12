from pathlib import Path
from city import City
from mo_tndp import MOTNDP

if __name__ == '__main__':
    dilemma_dir = Path(f"./cities/dilemma_5x5")
    city = City(
            dilemma_dir, 
            groups_file="groups.txt",
            ignore_existing_lines=True
    )

    # arguments -- should be passed in from command line
    nr_episodes = 10
    nr_stations = 9 # essentially steps in the episode
    seed = 42
    
    env = MOTNDP(city, nr_stations)
    training_step = 0
    for _ in range(nr_episodes):
        state = env.reset(seed=seed)
        while True:
            # Implement the agent policy here
            action = env.action_space.sample()
            new_state, reward, done = env.step(action)

            training_step += 1
            print(f'step {training_step}, state: {state}, action: {action}, reward: {reward} new_state: {new_state}')

            if done:
                break

            state = new_state

    print('all done')