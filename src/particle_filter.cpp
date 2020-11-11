/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine gen;

constexpr auto init_weight{1.0};

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  particles.reserve(num_particles);
  weights.resize(num_particles);

  for (int id = 1; id <= num_particles; ++id) {
    particles.emplace_back(
        Particle{id, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight});
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

  static const auto update_particles_v1 = [&](Particle& p) {
    const double th0 = p.theta;
    const double dx = delta_t * velocity * cos(th0);
    const double dy = delta_t * velocity * sin(th0);
    p.x = p.x + dx + dist_x(gen);
    p.y = p.y + dy + dist_y(gen);
  };

  static const auto update_particles_v2 = [&](Particle& p) {
    const double th0 = p.theta;
    const double dth = yaw_rate * delta_t;
    const double dx = velocity * (sin(th0 + dth) - sin(th0)) / yaw_rate;
    const double dy = velocity * (cos(th0) - cos(th0 + dth)) / yaw_rate;
    p.x = p.x + dx + dist_x(gen);
    p.y = p.y + dy + dist_y(gen);
    p.theta = th0 + dth + dist_theta(gen);
  };

  if (abs(yaw_rate) < 1E-6) {
    std::for_each(particles.begin(), particles.end(), update_particles_v1);
  } else {
    std::for_each(particles.begin(), particles.end(), update_particles_v2);
  }
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs>& predicted,
                                     vector<LandmarkObs>& observations) {
  // no modification of observation id if there's no updates
  for (auto& observation : observations) {
    double nearest_distance = std::numeric_limits<double>::max();
    int nearest_id = 0;
    for (const auto& landmark : predicted) {
      const double distance =
          dist(observation.x, observation.y, landmark.x, landmark.y);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_id = landmark.id;
      }
    }
    observation.id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  const double sigma_x = std_landmark[0];
  const double sigma_y = std_landmark[1];
  const double sigma_xx = sigma_x * sigma_x;
  const double sigma_yy = sigma_y * sigma_y;
  const double sigma_xx_inv = 1.0 / sigma_xx;
  const double sigma_yy_inv = 1.0 / sigma_yy;
  const double gaussian_norm = 1.0 / (2 * M_PI * sigma_x * sigma_y);

  // will not update if there are no observations
  if (observations.empty()) {
    return;
  }

  int i = -1;
  for (auto& particle : particles) {
    i += 1;
    const double p_x = particle.x;
    const double p_y = particle.y;
    const double p_theta = particle.theta;

    // Create list of predicted landmarks in sensor range (/map frame).
    vector<LandmarkObs> predicted_observations =
        getNearLandmarks(p_x, p_y, sensor_range, map_landmarks);

    if (predicted_observations.empty()) {
      // dont update this particle and assign low weight;
      particle.weight = 1e-9;
      weights[i] = particle.weight;
      continue;
    }

    auto map_observations = std::vector<LandmarkObs>{observations};

    auto get_map_coordinates = [&](LandmarkObs& observation) {
      double map_obs_x =
          p_x + cos(p_theta) * observation.x - sin(p_theta) * observation.y;
      double map_obs_y =
          p_y + sin(p_theta) * observation.x + cos(p_theta) * observation.y;

      observation.x = map_obs_x;
      observation.y = map_obs_y;
    };

    std::for_each(map_observations.begin(), map_observations.end(),
                  get_map_coordinates);

    // associate observations to given landmarks.
    dataAssociation(predicted_observations, map_observations);

    // std::vector<int> associations;
    // std::vector<double> sense_x;
    // std::vector<double> sense_y;

    // update weight using multivariate gaussian distribution
    particle.weight = 1;
    for (const auto& observation : map_observations) {
      // debugging
      // associations.push_back(observation.id);
      // sense_x.push_back(observation.x);
      // sense_y.push_back(observation.y);

      // get associated landmark
      LandmarkObs landmark;
      if (observation.id >= 0 &&
          getLandmarkById(predicted_observations, observation.id, landmark)) {
        // compute local weight
        const double delta_x = observation.x - landmark.x;
        const double delta_y = observation.y - landmark.y;
        const double local_weight =
            gaussian_norm * exp(-0.5 * (delta_x * delta_x * sigma_xx_inv +
                                        delta_y * delta_y * sigma_yy_inv));
        particle.weight *= local_weight;
        weights[i] = particle.weight;
      }
    }
    // SetAssociations(particle, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  std::discrete_distribution<> sampler(weights.begin(), weights.end());

  std::vector<Particle> new_particles{num_particles};

  std::generate(new_particles.begin(), new_particles.end(),
                [&]() { return particles[sampler(gen)]; });

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}