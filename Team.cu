#include "Team.cuh"

Team::Team(int id, float athlete_speeds[3]) : team_id(id) {
    athletes.resize(3);  // Allocate space for 3 athletes
    for (int i = 0; i < 3; ++i) {
        athletes[i] = Athlete(i, id, athlete_speeds[i]);
    }
    std::cout << "Athletes are created in team." << std::endl;
    std::cout << "Athletes in team: " << athletes.size() << std::endl;
}

int Team::getTeamId() {
    return this->team_id;
}

void Team::setTeamId(int team_id) {
    this->team_id = team_id;
}

std::vector<Athlete> Team::getAthletes() {
    return this->athletes;
}