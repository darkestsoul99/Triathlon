#include "Race.cuh"
#include "Athlete.cuh"
#include <iostream>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thread>
#include <chrono>

#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel to update athlete positions
__global__ void updatePositions(Athlete* athletes, float raceTime) {
    int segment_distances[3] = { 5000, 45000, 100000 }; // Swimming, Cycling, Running distances
    int num_athletes = 900;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_athletes; i += stride) {
        if (athletes[i].race_finished == false) {
            // Update athlete's position
            athletes[i].position += athletes[i].speed;
            athletes[i].time += 1; // 1 second per update

            // Handle segment transitions
            if (athletes[i].segment == 0 && athletes[i].position >= segment_distances[0]) {
                athletes[i].speed *= 3;
                athletes[i].time += 10;
                athletes[i].segment = 1;
                athletes[i].position = segment_distances[0]; // Exact segment boundary
            }
            else if (athletes[i].segment == 1 && athletes[i].position >= segment_distances[1]) {
                athletes[i].speed /= 3;
                athletes[i].time += 10;
                athletes[i].segment = 2;
                athletes[i].position = segment_distances[1]; // Exact segment boundary
            }
            else if (athletes[i].segment == 2 && athletes[i].position >= segment_distances[2]) {
                athletes[i].time += raceTime;
                athletes[i].position = segment_distances[2];
                athletes[i].race_finished = true;
            }
        }
    }
}

Race::Race(int n, std::vector<std::vector<float>>& athlete_speeds) : num_teams(n), raceTime(0.0) {
    for (int i = 0; i < num_teams; ++i) {
        teams.emplace_back(i, athlete_speeds[i].data());
    }
    std::cout << "Race created." << std::endl;
    std::cout << "Number of teams: " << teams.size() << std::endl;
}

cudaError_t Race::startRace(const int team_index, const int athlete_index) {
    std::cout << "Race started." << std::endl;
    int num_athletes = num_teams * 3;
    Athlete* athletes;
    int segment_distances[3] = { 5000, 45000, 55000 }; // Swimming, Cycling, Running distances
    cudaError_t cudaStatus;
    bool firstAthlete = false;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    // Allocate managed memory for athletes
    gpuErrchk(cudaStatus = cudaMallocManaged(&athletes, num_athletes * sizeof(Athlete)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed!");
        goto Error;
    }

    // Initialize athlete data
    for (int i = 0; i < num_teams; ++i) {
        for (int j = 0; j < 3; ++j) {
            int idx = i * 3 + j;
            athletes[idx].team_id = i;
            athletes[idx].speed = teams[i].athletes[j].speed;
            athletes[idx].position = teams[i].athletes[j].position; // Start at the beginning
            athletes[idx].time = teams[i].athletes[j].time; // Start time
            athletes[idx].segment = teams[i].athletes[j].segment; // Start in swimming segment
        }
    }

    while (true) {
        int num_blocks = static_cast<int>((num_athletes + 255) / 256);
        int blockSize = 256;
        updatePositions<<<num_blocks, blockSize>>>(athletes, raceTime);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "updatePositions launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updatePositions!\n", cudaStatus);
            goto Error;
        }

        // Check for race completion
        bool race_ongoing = false;
        for (int i = 0; i < num_athletes; ++i) {
            if (athletes[i].position < segment_distances[2]) {
                race_ongoing = true;
                break;
            }
            else {
                if (firstAthlete == false) {
                    printAthleteResults(athletes);
                    firstAthlete = true;
                }
            }
        }

        // Sleep for a second to simulate real-time updates (if running on a system where this is feasible)
        std::this_thread::sleep_for(std::chrono::seconds(1));
        raceTime++;
        // Print debugging information
        int i = team_index * 3 + athlete_index;
        printf("Athlete %d: Position = %f, Speed = %f, Time = %f, Segment = %d\n", i, athletes[i].position, athletes[i].speed, athletes[i].time, athletes[i].segment);
        if (!race_ongoing) break;
    }

    // Print final results
    printTeamResults(athletes);

Error:
    // Free managed memory
    gpuErrchk(cudaFree(athletes));

    return cudaStatus;
}

void Race::printAthleteResults(Athlete* athletes) {
    std::cout << "First Winner has finished the triathlon. Current results :\n";
    // Print individual athlete results
    for (int i = 0; i < num_teams; ++i) {
        for (int j = 0; j < 3; ++j) {
            int idx = i * 3 + j;
            std::cout << "Athlete in team " << teams[i].team_id << " - Position: " << athletes[idx].position
                << ", Speed: " << athletes[idx].speed << ", Time: " << athletes[idx].time << " seconds\n";
        }
    }
}

void Race::printTeamResults(Athlete* athletes) {
    // Calculate team rankings based on total time
    std::vector<std::pair<int, float>> team_times;
    for (int i = 0; i < num_teams; ++i) {
        float total_time = 0.0f;
        for (int j = 0; j < 3; ++j) {
            int idx = i * 3 + j;
            total_time += athletes[idx].time;
        }
        team_times.emplace_back(i, total_time);
    }

    // Sort teams by total time
    std::sort(team_times.begin(), team_times.end(), [](const auto& left, const auto& right) {
        return left.second < right.second;
        });

    // Print team rankings
    std::cout << "Team Rankings:\n";
    for (size_t i = 0; i < team_times.size(); ++i) {
        std::cout << "Rank " << (i + 1) << ": Team " << team_times[i].first
            << " Total Time: " << team_times[i].second << " seconds\n";
    }
}
