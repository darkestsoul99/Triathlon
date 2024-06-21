#ifndef RACE_CUH
#define RACE_CUH

#include <vector>
#include <iostream>
#include "Team.cuh"
#include "Athlete.cuh"


class Race {
public:
    std::vector<Team> teams;
    int num_teams;
    float raceTime;

    Race(int n, std::vector<std::vector<float>>& athlete_speeds);
    cudaError_t startRace(const int team_index, const int athlete_index);
    void printAthleteResults(Athlete* athletes);
    void printTeamResults(Athlete* athletes);
};

#endif // RACE_CUH
