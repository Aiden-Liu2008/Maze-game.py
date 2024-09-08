import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time
from datetime import timedelta

MAZE_HEIGHT = 21
MAZE_WIDTH = 41


# Maze generation using DFS algorithm
def generate_maze(height, width):
    maze = [[1 for _ in range(width)] for _ in range(height)]
    stack = []

    # Create an entrance and exit
    entrance = random.randint(1, width - 2)
    exit = random.randint(1, width - 2)
    maze[0][entrance] = 0
    maze[height - 1][exit] = 0

    # DFS algorithm
    def dfs(x, y):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < height and 0 <= new_y < width and maze[new_x][new_y] == 1:
                maze[x + dx // 2][y + dy // 2] = 0
                maze[new_x][new_y] = 0
                stack.append((new_x, new_y))
                dfs(new_x, new_y)

    # Start generating from the entrance
    start_x, start_y = 0, entrance
    stack.append((start_x, start_y))
    dfs(start_x, start_y)

    # Set end point
    end_x, end_y = height - 1, exit
    maze[end_x][end_y] = '*'

    return maze, (end_x, end_y)


# Convert maze characters to numerical values
def convert_maze_to_state(maze, char_pos):
    state = np.zeros((len(maze), len(maze[0])), dtype=int)
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if [i, j] == char_pos:
                state[i][j] = 2  # Agent position
            elif cell == 0:
                state[i][j] = 0  # Open space
            elif cell == '*':
                state[i][j] = 3  # Goal
            else:
                state[i][j] = 1  # Wall
    return state


# Maze environment class with refresh level functionality
class MazeEnv:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.maze, self.end_pos = generate_maze(height, width)
        self.char_pos = [0, self.maze[0].index(0)]
        self.saved_maze = None  # For storing the current maze for refresh

    def reset(self, refresh=False):
        if refresh and self.saved_maze is not None:
            self.maze = [row[:] for row in self.saved_maze]  # Restore maze
        else:
            self.maze, self.end_pos = generate_maze(self.height, self.width)
            self.saved_maze = [row[:] for row in self.maze]  # Save the current maze
        self.char_pos = [0, self.maze[0].index(0)]
        return convert_maze_to_state(self.maze, self.char_pos)

    def step(self, action):
        move_map = {'w': (-1, 0), 'a': (0, -1), 's': (1, 0), 'd': (0, 1)}
        if action in move_map:
            new_pos = [self.char_pos[0] + move_map[action][0], self.char_pos[1] + move_map[action][1]]
            if self.is_within_bounds(new_pos) and self.maze[new_pos[0]][new_pos[1]] in [0, '*']:
                self.char_pos = new_pos
                if self.is_end_point():
                    return convert_maze_to_state(self.maze, self.char_pos), 1, True  # Reward, Done
                return convert_maze_to_state(self.maze, self.char_pos), -0.01, False  # Small penalty for each step
        return convert_maze_to_state(self.maze, self.char_pos), -0.1, False  # Penalty for invalid move

    def is_within_bounds(self, pos):
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width

    def is_end_point(self):
        return self.char_pos == list(self.end_pos)


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Progress bar with time estimation
def progress_bar(episode, total_episodes, start_time, bar_length=30):
    progress = episode / total_episodes
    elapsed_time = time.time() - start_time
    time_str = f"Elapsed: {str(timedelta(seconds=int(elapsed_time)))}"
    return f"\rLevel {episode + 1}/{total_episodes} | {time_str}"


# Training function with progress bar and refresh
def train(env, model, episodes, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64):
    optimizer = optim.Adam(model.parameters())
    memory = deque(maxlen=2000)
    action_map = {'w': 0, 'a': 1, 's': 2, 'd': 3}

    start_time = time.time()
    last_update_time = start_time
    for episode in range(episodes):
        state = env.reset()  # Start with a new maze for each episode
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)  # Convert to Tensor
        total_reward = 0

        for t in range(1000):
            # Check if it's time to update progress (every minute)
            if time.time() - last_update_time >= 60:
                print(progress_bar(episode + 1, episodes, start_time), end='')
                last_update_time = time.time()

            # Select action: exploration vs exploitation
            if random.random() < epsilon:
                action = random.choice(['w', 'a', 's', 'd'])
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = ['w', 'a', 's', 'd'][q_values.argmax().item()]

            # Step in the environment
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
            memory.append((state, action_map[action], reward, next_state, done))
            state = next_state
            total_reward += reward

            if done:
                print(f"\nLevel {episode + 1}/{episodes}, Total Reward: {total_reward}")
                break

            # Training the model if enough memory is available
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                next_states = torch.cat(next_states)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)

                q_values = model(states)
                next_q_values = model(next_states)

                q_target = rewards + gamma * next_q_values.max(1)[0] * (1 - dones)
                q_target = q_target.unsqueeze(1)

                action_indices = torch.LongTensor(actions)
                loss = F.mse_loss(q_values.gather(1, action_indices.unsqueeze(1)), q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # Save model after training
    torch.save(model.state_dict(), 'dqn_model.pth')


# Run training
env = MazeEnv(MAZE_HEIGHT, MAZE_WIDTH)
input_dim = MAZE_HEIGHT * MAZE_WIDTH
output_dim = 4  # Number of actions
model = DQN(input_dim, output_dim)
train(env, model, episodes=100)
