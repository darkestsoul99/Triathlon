#include "Athlete.cuh"
#include <iostream>

void Athlete::initialize(int id, int team_id, float initial_speed) {
    this->id = id;
    this->team_id = team_id;
    this->position = 0;
    this->speed = initial_speed;
    this->segment = 0;
    this->time = 0.0f;
    this->race_finished = false;
    std::cout << "Speed of athlete " << this->id << " of team " << this->team_id << " : " << this->speed << std::endl;
}