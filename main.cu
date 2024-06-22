#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Race.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <chrono>


int main(int argc, char** argv)
{
    const int num_teams = 300;
    std::vector<std::vector<float>> athlete_speeds(num_teams, std::vector<float>(3));

    // Initialize random speeds for athletes
    std::default_random_engine gen(std::time(nullptr));
    std::uniform_real_distribution<float> distrib(1.0f, 5.0f);

    // Set cout to display floating-point numbers with 2 digits after the decimal point
    std::cout << std::fixed << std::setprecision(2);

    for (int i = 0; i < num_teams; ++i) {
        for (int j = 0; j < 3; ++j) {
            athlete_speeds[i][j] = distrib(gen);
        }
    }

    Race race(num_teams, athlete_speeds);

    // Check if arguments are provided correctly
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <team_index> <athlete_index>" << std::endl;
        return 1;
    }

    int team_index = std::atoi(argv[1]);
    int athlete_index = std::atoi(argv[2]);

    // Validate input indices
    if (team_index < 0 || team_index >= num_teams) {
        std::cerr << "Team index out of range. Should be between 0 and " << num_teams - 1 << std::endl;
        return 1;
    }

    if (athlete_index < 0 || athlete_index >= 3) {
        std::cerr << "Athlete index out of range. Should be between 0 and 2" << std::endl;
        return 1;
    }
    
    // Start the race with passing tracking info of specific athlete
    cudaError_t cudaStatus = race.startRace(team_index, athlete_index);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to start race!\n");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    return 0;
}
