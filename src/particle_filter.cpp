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

  for (int id = 1; id <= num_particles; ++id) {
    particles.emplace_back(
        Particle{id, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight});
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
 std::default_random_engine gen;

  auto x_std = std_pos[0];
  auto y_std = std_pos[1];
  auto yaw_std = std_pos[2];

  double new_x, new_y, new_theta;

  for (uint i = 0; i < num_particles; i++)
  {
    if (yaw_rate == 0)
    {
      new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      new_theta = particles[i].theta;
    }
    else
    {
      new_x = particles[i].x +
              (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      new_y = particles[i].y +
              (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      new_theta = particles[i].theta + yaw_rate * delta_t;
    }
    std::normal_distribution<double> dist_x(new_x, x_std);
    particles[i].x = dist_x(gen);
    std::normal_distribution<double> dist_y(new_y, y_std);
    particles[i].y = dist_y(gen);
    std::normal_distribution<double> dist_theta(new_theta, yaw_std);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs>& range_landmarks,
                                     vector<LandmarkObs>& observations) {
  // no modification of observation id if there's no updates
  for (auto& observation : observations) {
    double nearest_distance =  dist(observation.x, observation.y, range_landmarks[0].x, range_landmarks[0].y);

    int nearest_id = 0;
    
    for (size_t i = 1; i < range_landmarks.size(); ++i) {
      auto landmark = range_landmarks[i];
      double distance =
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


  const auto gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  const auto gauss_den_x = 2 * pow(std_landmark[0], 2);
  const auto gauss_den_y = 2 * pow(std_landmark[1], 2);

  if (observations.empty()) {
    return;
  }

  for (int i = 0; i < num_particles; ++i) {
    const double& p_x = particles[i].x;
    const double& p_y = particles[i].y;
    const double& p_sin = sin(particles[i].theta);
    const double& p_cos = cos(particles[i].theta);

    // Create list of landmarks in sensor range
    vector<LandmarkObs> landmarks_in_range =
        get_landmarks_close_to_particle(p_x, p_y, sensor_range, map_landmarks);

    if (landmarks_in_range.empty()) {
      particles[i].weight = 1e-3;
      weights[i] =  1e-3;

      continue;
    }

    // Project observation coordinates into map coordinates
    auto map_observations = std::vector<LandmarkObs>{observations};

    auto get_map_coordinates = [&](LandmarkObs& observation) {
      double map_obs_x = p_x + p_cos * observation.x - p_sin * observation.y;
      double map_obs_y = p_y + p_sin * observation.x + p_cos * observation.y;

      observation.x = map_obs_x;
      observation.y = map_obs_y;
    };

    std::for_each(map_observations.begin(), map_observations.end(),
                  get_map_coordinates);

    // Map each observation with the closest landmark to the particle
    dataAssociation(landmarks_in_range, map_observations);

    // For each observation calculate the distance to the closest landmark, the
    // productory of all distances is the weight of the particle
    double W = init_weight;
    for (const auto& observation : map_observations) {
      auto nearest_landmark = landmarks_in_range.at(observation.id);

      auto exponent = pow(observation.x - nearest_landmark.x, 2) / gauss_den_x + pow(observation.y - nearest_landmark.y, 2) / gauss_den_y;
      W *= gauss_norm * exp(-exponent);

    }

    particles[i].weight = W;
    weights[i] = W;
    // SetAssociations(particles[i], associations, sense_x, sense_y);
  }

  // normalize
  // double sum_w = accumulate(weights.begin(), weights.end(), 0.0);
  // std::for_each(weights.begin(), weights.end(),
  //              [&](double x) { return x / sum_w; });
}

void ParticleFilter::resample() {
  vector<Particle> new_particles(num_particles);

  double beta = 0;
  int index = rand() % num_particles;
  auto max_weight_element = max_element(weights.begin(), weights.end());

  for (uint i = 0; i < num_particles; ++i)
  {

    beta += (rand() / (RAND_MAX + 1.0)) * (2 * (*max_weight_element));
    while (weights[index] < beta)
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles[i] = particles[index];
  }
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