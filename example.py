from pathlib import Path
from motndp.city import City
import gymnasium
from gymnasium.envs.registration import register

from motndp.constraints import BasicConstraints, MetroConstraints

register(
    id="motndp_dilemma-v0",
    entry_point="motndp.motndp:MOTNDP"
)

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
    
    env = gymnasium.make('motndp_dilemma-v0', city=city, constraints=MetroConstraints(city), nr_stations=nr_stations)
    training_step = 0
    for _ in range(nr_episodes):
        state, info = env.reset(seed=seed)
        while True:
            # Implement the agent policy here
            action = env.action_space.sample(mask=info['action_mask'])
            new_state, reward, done, _, info = env.step(action)

            training_step += 1
            print(f'step {training_step}, state: {state["location"]}, action: {action}, reward: {reward} new_state: {new_state["location"]}')

            if done:
                break

            state = new_state
