import random
import os
import sys

MAZE_HEIGHT = 21
MAZE_WIDTH = 41

#position
char_pos = [0, 0]  #starts at the entrance

#Maze generation using recursive backtracking algorithm
def generate_maze(height, width):
    maze = [[1 for _ in range(width)] for _ in range(height)]
    stack = []

    #Create an entrance, 1 top 1 bottom
    entrance = random.randint(1, width-2)
    maze[0][entrance] = 0
    maze[height-1][random.randint(1, width-2)] = 0

    #Recursive backtracking algorithm
    def backtrack(x, y):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            new_x, new_y = x + 2 * dx, y + 2 * dy

            if 0 <= new_x < height and 0 <= new_y < width and maze[new_x][new_y] == 1:
                maze[x + dx][y + dy] = 0
                maze[new_x][new_y] = 0
                stack.append((new_x, new_y))
                backtrack(new_x, new_y)

    # Start generating - ramdom mazzzzzzz
    start_x, start_y = random.randint(0, height-1), random.randint(0, width-1)
    backtrack(start_x, start_y)

    return maze

# Print maze
def print_maze(maze, char_pos):
    os.system('clear') if os.name == 'posix' else os.system('cls')
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if [i, j] == char_pos:
                print("P", end="")
            elif maze[i][j] == 0:
                print(" ", end="")
            else:
                # Print a pattern for walls
                print("█", end="")
        print()

# Check if a position is within the maze boundaries
def is_within_bounds(pos, maze_height, maze_width):
    return 0 <= pos[0] < maze_height and 0 <= pos[1] < maze_width

# Main game loop
def main():
    maze = generate_maze(MAZE_HEIGHT, MAZE_WIDTH)

    # Set character position to the entrance
    char_pos[0] = 0
    char_pos[1] = maze[0].index(0)

    while True:
        print_maze(maze, char_pos)
        move = input("Enter WASD to move (Q to quit): ").lower()
        if move == 'q':
            break
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

            if is_within_bounds(new_char_pos, MAZE_HEIGHT, MAZE_WIDTH) and maze[new_char_pos[0]][new_char_pos[1]] == 0:
                char_pos[:] = new_char_pos

if __name__ == "__main__":
    main()
# Done
