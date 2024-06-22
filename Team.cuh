#pragma once
#ifndef TEAM_CUH
#define TEAM_CUH

#include <vector>
#include <iostream>
#include "Athlete.cuh"

class Team {
private:
    // Team attributes
    int team_id;
    std::vector<Athlete> athletes;
public:
    // Getters and Setters
    int getTeamId();
    void setTeamId(int team_id);
    std::vector<Athlete> getAthletes();

    // Team Constructor
    Team(int id, float athlete_speeds[3]);

};

#endif // TEAM_CUH


