#ifndef ATHLETE_CUH
#define ATHLETE_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class Athlete {
private:
    int id;
    int team_id;
    float position;
    float speed;
    int segment;
    float finishTime;
    bool race_finished;

public:
    __host__ __device__ __forceinline__ int getId() const;
    __host__ __device__ __forceinline__ void setId(int id);
    __host__ __device__ __forceinline__ int getTeamId() const;
    __host__ __device__ __forceinline__ void setTeamId(int team_id);
    __host__ __device__ __forceinline__ float getPosition() const;
    __host__ __device__ __forceinline__ void setPosition(float position);
    __host__ __device__ __forceinline__ float getSpeed() const;
    __host__ __device__ __forceinline__ void setSpeed(float speed);
    __host__ __device__ __forceinline__ int getSegment() const;
    __host__ __device__ __forceinline__ void setSegment(int segment);
    __host__ __device__ __forceinline__ float getFinishTime() const;
    __host__ __device__ __forceinline__ void setFinishTime(float finishTime);
    __host__ __device__ __forceinline__ bool getRaceFinished() const;
    __host__ __device__ __forceinline__ void setRaceFinished(bool race_finished);

    // Athlete Constructor
    __host__ __device__ Athlete(int id = 0, int team_id = 0, float initial_speed = 0.0f);
};

#endif // ATHLETE_CUH

