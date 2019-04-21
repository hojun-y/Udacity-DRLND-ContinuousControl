import Agent.agent as ddpg_agent
from Agent.memory_utils import StateBuilder
from config import config
from collections import deque
import Agent.plot_utils as plotter
from unityagents import UnityEnvironment
import network
import numpy as np
import pickle

agent = ddpg_agent.DDQNAgent(network.DDPGActor, network.DDPGCritic, config)
state_builder = StateBuilder(config['history_len'])

env = UnityEnvironment(config['env_path'])
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

episodes = 0
steps = 0
total_scores = deque(maxlen=100)
scores = []
train_flag = True
loss = None
while train_flag:
    try:
        env_info = env.reset(train_mode=True)[brain_name]
        observation = env_info.vector_observations[0]
        state_builder.reset(observation)
        state = state_builder.get_state()

        total_score = 0
        done = False
        while not done:
            if steps < config['train_start']:
                action = np.random.uniform(config['action_low'], config['action_high'], 4)
            else:
                action = agent.get_action(state)
            env_info = env.step(action)[brain_name]
            next_observation = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            state_builder.append(next_observation)
            next_state = state_builder.get_state()

            agent.append_memory(state, action, reward, next_state, done)
            state = next_state

            total_score += reward
            steps += 1

            if steps > config['train_start']:
                loss = agent.train()
                agent.soft_update(config['tau'])

            if done:
                episodes += 1
                total_scores.append(total_score)
                scores.append(total_score)
                if np.mean(total_scores) > config['target_score']:
                    train_flag = False
                    env.close()

        if episodes % config['print_every'] == 0:
            print('Episode {}'.format(episodes),
                  "\tStep: {}".format(steps),
                  "\tScore: {:.4f}".format(total_score), end="")
            if loss is not None:
                print("\tLoss: {:.4f}".format(loss))
            else:
                print("\t[RANDOM ACTION]")

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Saving model...")
        train_flag = False
        env.close()

# Save data
agent.save_weights(config['weights_save_path'])
with open(config['rewards_save_path'], 'wb') as f:
    pickle.dump(scores, f)
    f.close()

plotter.save_line_plot(scores, "Total Reward / Episode", config['plot_save_path'])
