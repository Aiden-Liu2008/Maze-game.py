import random
import os
import time
import numpy as np
import torch
import threading

MAZE_HEIGHT = 21
MAZE_WIDTH = 41

# Position
char_pos = [0, 0]  # Starts at the entrance
level_times = []  # List to record time taken for each level
stop_event = threading.Event()

# Timer function
def timer(seconds):
    time.sleep(seconds)
    stop_event.set()

# Maze generation using DFS algorithm
def generate_maze(height, width):
    maze = [[1 for _ in range(width)] for _ in range(height)]
    stack = []

    # Create an entrance and exit
    entrance = random.randint(1, width-2)
    exit = random.randint(1, width-2)
    maze[0][entrance] = 0
    maze[height-1][exit] = 0

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

# Print maze
def print_maze(maze, char_pos, level):
    os.system('clear') if os.name == 'posix' else os.system('cls')
    
    # ASCII pattern
    pattern = "###" * (MAZE_WIDTH // 3)
    print(pattern)
    print(f"{' ' * ((MAZE_WIDTH - len(f'Level {level}')) // 2)}Level {level}")
    print(pattern)
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if [i, j] == char_pos:
                print("P", end="")
            elif maze[i][j] == 0:
                print(" ", end="")
            elif maze[i][j] == '*':
                print("*", end="")
            else:
                print("█", end="")
        print()
    
    print(pattern)
    print(f"{' ' * ((MAZE_WIDTH - len(f'Level {level}')) // 2)}Level {level}")
    print(pattern)

# Check if a position is within the maze boundaries
def is_within_bounds(pos, maze_height, maze_width):
    return 0 <= pos[0] < maze_height and 0 <= pos[1] < maze_width

# Check if the character has reached the end point
def is_end_point(char_pos, end_pos):
    return char_pos == list(end_pos)

# A* Pathfinding Algorithm
def a_star(maze, start, end):
    open_set = {tuple(start)}
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): heuristic(start, end)}

    while open_set:
        current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
        
        if current == tuple(end):
            return reconstruct_path(came_from, current)

        open_set.remove(current)
        for neighbor in get_neighbors(current, maze):
            tentative_g_score = g_score[current] + 1
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                open_set.add(neighbor)

    return []

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, maze):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        neighbor = (pos[0] + dx, pos[1] + dy)
        if is_within_bounds(neighbor, MAZE_HEIGHT, MAZE_WIDTH) and maze[neighbor[0]][neighbor[1]] in [0, '*']:
            neighbors.append(neighbor)
    return neighbors

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# Enhanced Progress Bar with Tibetan Symbols
def progress_bar(iteration, total, length=40):
    percent = (iteration / total)
    bar_length = int(length * percent)
    bar = '༄' * bar_length + '─' * (length - bar_length)  # Tibetan symbol for a more fancy look
    print(f'\r|{bar}| {percent:.2%}', end='')

# Main game loop with AI
def main():
    duration = int(input("Enter the duration (in seconds) for the AI to run: "))
    timer_thread = threading.Thread(target=timer, args=(duration,))
    timer_thread.start()

    level = 1
    while not stop_event.is_set():
        maze, end_pos = generate_maze(MAZE_HEIGHT, MAZE_WIDTH)

        # Set character position to the entrance
        char_pos[0] = 0
        char_pos[1] = maze[0].index(0)

        start_time = time.time()  # Start the timer for the level

        while not stop_event.is_set():
            print_maze(maze, char_pos, level)
            path = a_star(maze, char_pos, end_pos)

            if not path:
                print("No path found, refreshing maze...")
                time.sleep(1)
                break  # Break out of the inner loop to regenerate the maze

            for step in path[1:]:  # Skip the starting position
                char_pos[:] = list(step)
                print_maze(maze, char_pos, level)
                progress_bar(path.index(step), len(path))
                time.sleep(0.03)  # Reduced time to simulate faster input key speed

            if is_end_point(char_pos, end_pos):
                end_time = time.time()  # End the timer for the level
                level_time = end_time - start_time
                level_times.append(level_time)  # Record the time taken for this level
                print(f"\nLevel {level} complete in {level_time:.2f} seconds!")
                time.sleep(1)  # Auto wait before moving to the next level
                level += 1
                break

if __name__ == "__main__":
    main()
