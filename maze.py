import random
import os

MAZE_HEIGHT = 21
MAZE_WIDTH = 41

# Position
char_pos = [0, 0]  # Starts at the entrance

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
def print_maze(maze, char_pos):
    os.system('clear') if os.name == 'posix' else os.system('cls')
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

# Check if a position is within the maze boundaries
def is_within_bounds(pos, maze_height, maze_width):
    return 0 <= pos[0] < maze_height and 0 <= pos[1] < maze_width

# Check if the character has reached the end point
def is_end_point(char_pos, end_pos):
    return char_pos == list(end_pos)

# Main game loop
def main():
    level = 1
    while True:
        maze, end_pos = generate_maze(MAZE_HEIGHT, MAZE_WIDTH)

        # Set character position to the entrance
        char_pos[0] = 0
        char_pos[1] = maze[0].index(0)

        while True:
            print_maze(maze, char_pos)
            move = input("Enter WASD to move, R to refresh (Q to quit): ").lower()
            if move == 'q':
                return
            elif move == 'r':
                break  # Break out of the inner loop to regenerate the maze
            elif move in ['w', 'a', 's', 'd']:
                new_char_pos = char_pos[:]
                if move == 'w':
                    new_char_pos[0] -= 1
                elif move == 'a':
                    new_char_pos[1] -= 1
                elif move == 's':
                    new_char_pos[0] += 1
                elif move == 'd':
                    new_char_pos[1] += 1

                if is_within_bounds(new_char_pos, MAZE_HEIGHT, MAZE_WIDTH) and maze[new_char_pos[0]][new_char_pos[1]] in [0, '*']:
                    char_pos[:] = new_char_pos

                if is_end_point(char_pos, end_pos):
                    print_maze(maze, char_pos)
                    print(f"Level {level} complete!")
                    input("Press Enter to continue to the next level...")
                    level += 1
                    break

if __name__ == "__main__":
    main()