#pragma once
#ifndef TEAM_H
#define TEAM_H

#include <vector>
#include <iostream>
#include "Athlete.cuh"

class Team {
public:
    int team_id;
    std::vector<Athlete> athletes;

    Team(int id, float athlete_speeds[3]);
    float getTotalTime() const;
};

#endif // TEAM_H


