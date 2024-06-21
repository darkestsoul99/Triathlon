# Triathlon
# Triathlon Simulation Project

## Project Description

This project simulates a triathlon event where 300 teams, each consisting of 3 athletes, compete in swimming, cycling, and running. The goal is to track the position and speed of each athlete in real-time using CUDA C++ for parallel computation on a GPU. The simulation calculates the positions of the athletes every second and determines the winners based on individual and team performances.

## Features

- **Object-Oriented Programming (OOP):** The project is developed using OOP principles in C++ or CUDA C++.
- **Parallel Computation:** Athlete positions and speeds are computed in parallel using CUDA threads.
- **Real-time Simulation:** The simulation updates every second to reflect the real-time positions of all athletes.
- **Dynamic Speed Changes:** Athlete speeds change based on the event stage:
  - Swimming: Initial speed between 1 m/s and 5 m/s (randomly assigned).
  - Cycling: Speed triples.
  - Running: Speed reduces to one-third of the initial speed.
- **Transition Time:** Athletes lose 10 seconds during transitions between events.
- **User Input:** The program allows the user to specify which athlete(s) to track and display their speed and position during the race.
- **Completion:** The program prints the positions of all athletes when the first athlete finishes and displays team rankings when the race ends.

## Usage

### Prerequisites

- **Ubuntu Operating System:** The project is recommended to be run on Ubuntu.
- **CUDA Toolkit:** Ensure that CUDA is installed and properly configured.
- **C++ Compiler:** A C++ compiler with CUDA support is needed.

### Running the Simulation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/triathlon-simulation.git
   cd triathlon-simulation
2. **Compile the code:**
   ```bash
   nvcc -o triathlon triathlon.cu
3. **Run the simulation:**
   ```bash
   ./triathlon team_id athlete_id

## Example
- To track athletes with IDs 1, 2, and 3, run:
   ```bash
   ./triathlon team_id athlete_id

## Project Structure
- triathlon.cu: The main CUDA C++ source file containing the simulation logic.
- Makefile: Instructions for compiling the project.
- README.md: This README file.

## Output
- The positions and speeds of the specified athletes are printed at each second.
- When the first athlete finishes, the positions of all athletes are printed.
- After all athletes finish, the team rankings are displayed.

## Author
Berke Kocadere [berkekocadere@gmail.com]

## License
This project is licensed under the MIT License - see the - [MIT LICENSE](./LICENSE) file for details.

## Acknowledgements
- CUDA Toolkit Documentation
- Ubuntu Documentation
- Any other resources or individuals that contributed to the project.
