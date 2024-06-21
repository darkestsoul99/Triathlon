#ifndef ATHLETE_CUH
#define ATHLETE_CUH

#include <cuda_runtime.h>

class Athlete {
public:
    int id;
    int team_id;
    float position;
    float speed;
    int segment;
    float time;
    bool race_finished;

    void initialize(int id, int team_id, float initial_speed);
};

#endif // ATHLETE_CUH
