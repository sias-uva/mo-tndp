# This is a simple demo of how to use Q-learning to solve the TNDP, using the gymnasium framework.
# Here we assume a single reward function, which is the sum of the rewards of all groups.
# Thus, one can say the utility function is the (equal weighted) sum of the utilities of all groups.
from pathlib import Path
import random
import numpy as np
from motndp.city import City
import gymnasium
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt

from motndp.constraints import BasicConstraints, MetroConstraints

register(
    id="motndp_dilemma-v0",
    entry_point="motndp.motndp:MOTNDP"
)

alpha = 0.1 # learning rate
gamma = 0.2
epsilon = 1
max_epsilon = 1
min_epsilon = 0.00
decay = 0.02

train_episodes = 400
test_episodes = 1
nr_stations = 9
seed = 42

# follow pre-determined policy
policy = None
# Best policy
# policy = [0, 0, 0, 0, 2, 2, 2, 2]
# policy = [0, 1, 1, 0, 2, 2, 4, 6]

if __name__ == '__main__':
    dilemma_dir = Path(f"./cities/dilemma_5x5")
    city = City(
            dilemma_dir, 
            groups_file="groups.txt",
            ignore_existing_lines=True
    )
    
    env = gymnasium.make('motndp_dilemma-v0', city=city, constraints=MetroConstraints(city), nr_stations=nr_stations)

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    avg_rewards = []
    epsilons = []
    training_step = 0
    best_episode_reward = 0
    best_episode_segment = []
    for episode in range(train_episodes):
        state, info = env.reset(seed=seed, loc=(4, 0))
        episode_reward = 0
        episode_step = 0
        while True:            
            state_index = city.grid_to_vector(state['location'][None, :]).item()

            # Exploration-exploitation trade-off 
            exp_exp_tradeoff = random.uniform(0, 1)

            # follow predetermined policy (set above)
            if policy:
                action = policy[episode_step]
            # exploit
            elif exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state_index, :] * info['action_mask'])
            # explore
            else:
                action = env.action_space.sample(mask=info['action_mask'])
            
            new_state, reward, done, _, info = env.step(action)

            # Here we sum the reward to create a single-objective policy optimization
            reward = reward.sum()

            # Update Q-Table
            new_state_gid = city.grid_to_vector(new_state['location'][None, :]).item()
            Q[state_index, action] = Q[state_index, action] + alpha * (reward + gamma * np.max(Q[new_state_gid, :]) - Q[state_index, action])
            episode_reward += reward

            training_step += 1
            episode_step += 1
            print(f'step {training_step}, episode: {episode}, episode_step: {episode_step}, state: {state["location"]}, action: {action}, reward: {reward} new_state: {new_state["location"]}')

            state = new_state

            if done:
                break
        #Cutting down on exploration by reducing the epsilon 
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_episode_segment = info['segments']
        
        # Adding the total reward and reduced epsilon values
        rewards.append(episode_reward)
        # Save the average reward over the last 10 episodes
        avg_rewards.append(np.average(rewards[-10:]))
        epsilons.append(epsilon)

    #Visualizing results and total reward over all episodes
    x = range(train_episodes)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, rewards, label='rewards', color='lightgray')
    ax.plot(x, epsilons, label='epsilon', color='orange')
    ax.plot(x, avg_rewards, label='average reward', color='blue')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Training total reward')
    ax.set_ylim(0, None)
    ax.set_title('Total rewards over all episodes in training') 
    fig.legend()
    fig.savefig(f'{Path("./qlearning_results.png")}')

    # Testing the agent
    total_rewards = 0
    for episode in range(test_episodes):
        state, info = env.reset(seed=seed, loc=(4, 0))
        episode_reward = 0
        locations = []
        while True:
            state_index = city.grid_to_vector(state['location'][None, :]).item()
            locations.append(state['location'])
            action = np.argmax(Q[state_index,:] * info['action_mask'])
            new_state, reward, done, _, info = env.step(action)
            reward = reward.sum()
            episode_reward += reward      
            state = new_state    
            if done:
                break
        total_rewards += episode_reward

    print('Line Segments: ', locations)
