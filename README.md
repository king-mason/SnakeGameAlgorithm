# Snake Game Algorithm

A visualization of an algorithm solving the classic Snake game in Python. This project demonstrates how algorithms can be used to play and optimize game strategies automatically.

## Description

This project implements a bread-first search (BFS) algorithm applied to the classic Snake game, efficiently collecting food while avoiding collisions. The visualization allows you to observe the algorithm's decision-making process in real-time. This demonstrates the implementation of an algorithm to a real application. Like any application, there is still room for scenario-specific adaptations and improvements.

## Features

- BFS pathfinding and decision making
    - Guarantees the shortest path
- Maximum survival time in worst-case scenario (snake is trapped)
- Real-time visualization of algorithm
- Cohesive and functional coding design

## Demo



https://github.com/user-attachments/assets/edf4525d-db11-489b-9ed1-2ee1c6205e25



## Demo (max speed)



https://github.com/user-attachments/assets/c6cea66f-c53a-4a2c-9eaa-bcebd7b1eb18




## Project Structure

- `Main.py`: Runs Pygame framework and implements algorithms
- `Algorithm.py`: Contains the core search logic and functions
- `Testing.py`: Test cases for game functionality

## Future Improvements

Several potential enhancements could be made to improve the algorithm and visualization:

### Algorithm Optimizations
- Implement A* pathfinding algorithm for improving search efficiency
- Add predictive path planning to improve food collection routes
    - i.e. Make routes better for collecting future apples, less obstructive
- Develop heuristics for avoiding dead-end scenarios
    - Current problem it runs into at larger sizes on larger boards
- Implement machine learning approaches for optimization of path

### Visualization Enhancements
- Add step-by-step visualization of the pathfinding process
    - Show the search "thinking"
    - Add enable/disable feature
- Display performance metrics in real-time (path length, time metrics, shortest route length without obstructions)
- Implement speed controls while visualization is running
- Add GUI for customization controls

### Additional Options
- Implement benchmark tools for comparison of algorithm designs (or of different grid sizes)
- Add support for multiple food items simultaneously


## Requirements

- Python 3.x (3.8+ recommended)
```
pip install pygame
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/king-mason/SnakeGameAlgorithm.git
cd SnakeGameAlgorithm
```

2. Run the game:
```bash
python Main.py
```


## Contributing

Feel free to fork the repository and submit pull requests for any improvements. See the Future Improvements section for potential areas of enhancement.

## License

This project is open source and available under the [MIT License](LICENSE).

## Authors

Mason King (@king-mason)
Sameer Shenoy (@ShenoySameer)
