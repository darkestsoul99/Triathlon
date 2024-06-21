#include "Team.cuh"

Team::Team(int id, float athlete_speeds[3]) : team_id(id) {
    athletes.resize(3);  // Allocate space for 3 athletes
    for (int i = 0; i < 3; ++i) {
        athletes[i].initialize(i, id, athlete_speeds[i]);
    }
    std::cout << "Athletes are created in team." << std::endl;
    std::cout << "Athletes in team: " << athletes.size() << std::endl;
}

float Team::getTotalTime() const {
    float total_time = 0.0f;
    for (const auto& athlete : athletes) {
        total_time += athlete.time;
    }
    return total_time;
}
