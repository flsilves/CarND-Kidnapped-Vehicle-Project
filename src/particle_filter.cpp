/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>

#include <algorithm>
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

  for (auto id = 1u; id <= num_particles; ++id) {
    particles.emplace_back(Particle{static_cast<int>(id), dist_x(gen),
                                    dist_y(gen), dist_theta(gen), init_weight});
  }

  is_initialized = true;
}

static double get_dist(double mean, double std_dev) {
  std::normal_distribution<double> dist(mean, std_dev);
  return dist(gen);
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  const auto x_sigma{std_pos[0]};
  const auto y_sigma{std_pos[1]};
  const auto theta_sigma{std_pos[2]};

  static const auto update_particles_v1 = [&](Particle& p) {
    const double x = p.x + delta_t * velocity * cos(p.theta);
    const double y = p.y + delta_t * velocity * sin(p.theta);
    p.x = get_dist(x, x_sigma);
    p.y = get_dist(y, y_sigma);
    p.theta = get_dist(p.theta, theta_sigma);
  };

  static const auto update_particles_v2 = [&](Particle& p) {
    const double x =
        p.x + (velocity / yaw_rate) *
                  (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
    const double y =
        p.y + (velocity / yaw_rate) *
                  (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
    const double theta = p.theta + yaw_rate * delta_t;
    p.x = get_dist(x, x_sigma);
    p.y = get_dist(y, y_sigma);
    p.theta = get_dist(theta, theta_sigma);
  };

  if (fabs(yaw_rate) < 1e-5) {
    std::for_each(particles.begin(), particles.end(), update_particles_v1);
  } else {
    std::for_each(particles.begin(), particles.end(), update_particles_v2);
  }
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs>& predicted,
                                     vector<LandmarkObs>& observations) {
  for (auto& observation : observations) {
    double nearest_distance = std::numeric_limits<double>::max();
    int nearest_id = 0;
    for (auto i = 0u; i < predicted.size(); ++i) {
      auto landmark = predicted[i];
      const double distance =
          dist(observation.x, observation.y, landmark.x, landmark.y);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_id = i;
      }
    }
    observation.id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  const double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  const double var_x = std_landmark[0] * std_landmark[0];
  const double var_y = std_landmark[1] * std_landmark[1];

  if (observations.empty()) {
    return;
  }

  for (auto i = 0u; i < num_particles; weights[i] = particles[i].weight, ++i) {
    const double& p_x = particles[i].x;
    const double& p_y = particles[i].y;
    const double& p_theta = particles[i].theta;
    const double& p_sin = sin(p_theta);
    const double& p_cos = cos(p_theta);

    // Create list of landmarks in sensor range
    vector<LandmarkObs> landmarks_in_range =
        get_landmarks_close_to_particle(p_x, p_y, sensor_range, map_landmarks);

    if (landmarks_in_range.empty()) {
      particles[i].weight = 1e-9;
      continue;
    }

    // Project observation coordinates into map coordinates
    auto map_observations = std::vector<LandmarkObs>{observations};

    auto get_map_coordinates = [&](LandmarkObs& observation) {
      const double map_obs_x =
          p_x + p_cos * observation.x - p_sin * observation.y;
      const double map_obs_y =
          p_y + p_sin * observation.x + p_cos * observation.y;

      observation.x = map_obs_x;
      observation.y = map_obs_y;
    };

    std::for_each(map_observations.begin(), map_observations.end(),
                  get_map_coordinates);

    // Map each observation with the closest landmark to the particle
    dataAssociation(landmarks_in_range, map_observations);

    // For each observation calculate the distance to the closest landmark, the
    // productory of all distances is the weight of the particle
    double W{init_weight};
    for (const auto& observation : map_observations) {
      const auto nearest_landmark = landmarks_in_range.at(observation.id);

      const double delta_x = observation.x - nearest_landmark.x;
      const double delta_y = observation.y - nearest_landmark.y;

      const double b = delta_x * delta_x * 0.5 / var_x;
      const double c = delta_y * delta_y * 0.5 / var_y;
      W *= gauss_norm * exp(-(b + c));
    }

    particles[i].weight = W;
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