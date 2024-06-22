#ifndef RACE_CUH
#define RACE_CUH

#include <vector>
#include <iostream>
#include "Team.cuh"
#include "Athlete.cuh"
#include <algorithm>
#include <thread>
#include <chrono>

#include <stdio.h>


class Race {
private:
    std::vector<Team> teams;
    int num_teams;
    float raceTime;
public:
    // Getters and Setters
    std::vector<Team> getTeams();
    int getNumberOfTeams();
    void setNumberOfTeams(int num_teams);
    float getRaceTime();
    void setRaceTime(float raceTime);

    // Public functions
    cudaError_t startRace(const int team_index, const int athlete_index);
    void printAthleteResults(Athlete* athletes);
    void printTeamResults(Athlete* athletes);

    // Constructor
    Race(int n, std::vector<std::vector<float>>& athlete_speeds);
};

#endif // RACE_CUH
