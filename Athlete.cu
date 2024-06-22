#include "Athlete.cuh"
#include <iostream>


__host__ __device__ Athlete::Athlete(int id, int team_id, float initial_speed)
    : id(id), team_id(team_id), position(0.0f), speed(initial_speed), segment(0), finishTime(0.0f), race_finished(false) {
}

__host__ __device__ __forceinline__ int Athlete::getId() const {
    return this->id;
}

__host__ __device__ __forceinline__ void Athlete::setId(int id) {
    this->id = id;
}

__host__ __device__ __forceinline__ int Athlete::getTeamId() const {
    return this->team_id;
}

__host__ __device__ __forceinline__ void Athlete::setTeamId(int team_id) {
    this->team_id = team_id;
}

__host__ __device__ __forceinline__ float Athlete::getPosition() const {
    return this->position;
}

__host__ __device__ __forceinline__ void Athlete::setPosition(float position) {
    this->position = position;
}

__host__ __device__ __forceinline__ float Athlete::getSpeed() const {
    return this->speed;
}

__host__ __device__ __forceinline__ void Athlete::setSpeed(float speed) {
    this->speed = speed;
}

__host__ __device__ __forceinline__ int Athlete::getSegment() const {
    return this->segment;
}

__host__ __device__ __forceinline__ void Athlete::setSegment(int segment) {
    this->segment = segment;
}

__host__ __device__ __forceinline__ float Athlete::getFinishTime() const {
    return this->finishTime;
}

__host__ __device__ __forceinline__ void Athlete::setFinishTime(float finishTime) {
    this->finishTime = finishTime;
}

__host__ __device__ __forceinline__ bool Athlete::getRaceFinished() const {
    return this->race_finished;
}

__host__ __device__ __forceinline__ void Athlete::setRaceFinished(bool race_finished) {
    this->race_finished = race_finished;
}
