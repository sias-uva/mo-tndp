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
    
    env = MOTNDP(city)
    for _ in range(10):
        state = env.reset()
        for i in range(5):
            action = env.action_space.sample()
            new_state, reward, done = env.step(action)

            print(f'step {i}, state: {state}, action: {action}, new_state: {new_state}')

            state = new_state
        print(state)

    print('all good')