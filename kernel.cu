#include "Athlete.cuh"

// CUDA kernel to update athlete positions
__global__ void updatePositions(Athlete* athletes, float raceTime) {
    int segment_distances[3] = { 5000, 45000, 100000 }; // Swimming, Cycling, Running distances
    int num_athletes = 900;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_athletes; i += stride) {
        if (!athletes[i].getRaceFinished()) {
            // Update athlete's position
            athletes[i].setPosition(athletes[i].getPosition() + athletes[i].getSpeed());

            // Handle segment transitions
            if (athletes[i].getSegment() == 0 && athletes[i].getPosition() >= segment_distances[0]) {
                athletes[i].setSpeed(athletes[i].getSpeed() * 3);
                athletes[i].setFinishTime(athletes[i].getFinishTime() + 10);
                athletes[i].setSegment(1);
                athletes[i].setPosition(segment_distances[0]); // Exact segment boundary
            }
            else if (athletes[i].getSegment() == 1 && athletes[i].getPosition() >= segment_distances[1]) {
                athletes[i].setSpeed(athletes[i].getSpeed() / 3);
                athletes[i].setFinishTime(athletes[i].getFinishTime() + 10);
                athletes[i].setSegment(2);
                athletes[i].setPosition(segment_distances[1]); // Exact segment boundary
            }
            else if (athletes[i].getSegment() == 2 && athletes[i].getPosition() >= segment_distances[2]) {
                athletes[i].setFinishTime(athletes[i].getFinishTime() + raceTime);
                athletes[i].setPosition(segment_distances[2]);
                athletes[i].setRaceFinished(true);
            }
        }
    }
}