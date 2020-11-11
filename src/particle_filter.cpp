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
constexpr auto number_of_particles{200};

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  particles.reserve(number_of_particles);

  for (int id = 1; id <= number_of_particles; ++id) {
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

  static const auto update_particles_zero_yaw = [&](Particle& p) {
    const double dx = delta_t * velocity * cos(p.theta);
    const double dy = delta_t * velocity * sin(p.theta);
    p.x = p.x + dx + dist_x(gen);
    p.y = p.y + dy + dist_y(gen);
    return p;
  };

  static const auto update_particles = [&](Particle& p) {
    const double th0 = p.theta;
    const double dth = yaw_rate * delta_t;
    const double dx = velocity * (sin(th0 + dth) - sin(th0)) / yaw_rate;
    const double dy = velocity * (cos(th0) - cos(th0 + dth)) / yaw_rate;
    p.x = p.x + dx + dist_x(gen);
    p.y = p.y + dy + dist_y(gen);
    return p;
  };

  if (abs(yaw_rate) < 1E-6) {
    std::transform(particles.begin(), particles.end(), particles.begin(),
                   update_particles_zero_yaw);
  } else {
    std::transform(particles.begin(), particles.end(), particles.begin(),
                   update_particles);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  // no modification of observation id if there's no updates
  for (auto& observation : observations) {
    double nearest_distance = std::numeric_limits<double>::max();

    for (const auto& landmark : predicted) {
      const double distance =
          dist(observation.x, observation.y, landmark.x, landmark.y);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        observation.id = landmark.id;
      }
    }
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

  for (auto& particle : particles) {
    const double p_x = particle.x;
    const double p_y = particle.y;
    const double p_th = particle.theta;

    // Create list of predicted landmarks in sensor range (/map frame).
    vector<LandmarkObs> predicted_observations =
        getNearLandmarks(p_x, p_y, sensor_range, map_landmarks);
    if (predicted_observations.empty()) {
      // dont update this particle and assign low weight;
      particle.weight = 0.000001 / particles.size();
      continue;
    }

    // Transform observations from /car to /map frames.
    vector<LandmarkObs> map_observations;
    for (const auto& observation : observations) {
      map_observations.emplace_back(
          transformObservationToMap(observation, p_x, p_y, p_th));
    }

    // associate observations to given landmarks.
    dataAssociation(predicted_observations, map_observations);

    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    // update weight using multivariate gaussian distribution
    particle.weight = 1;
    for (const auto& observation : map_observations) {
      // debugging
      associations.push_back(observation.id);
      sense_x.push_back(observation.x);
      sense_y.push_back(observation.y);

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
      }
    }
    SetAssociations(particle, associations, sense_x, sense_y);
  }

  // normalize
  const double W =
      std::accumulate(particles.begin(), particles.end(), 0.0,
                      [](double r, const Particle& p) { return r + p.weight; });
  if (W < 0.000000001) {
    return;
  }
  for (auto& particle : particles) {
    particle.weight = particle.weight / W;
  }
  double alpha_sum = 0;
  for (const auto& particle : particles) {
    alpha_sum += particle.weight;
  }
}

void ParticleFilter::resample() {
  std::vector<double> weights;
  weights.reserve(particles.size());

  std::transform(particles.begin(), particles.end(),
                 std::back_inserter(weights),
                 [](Particle& p) { return p.weight; });

  // weighted discrete distribution in range [0, N)
  std::discrete_distribution<> d(weights.begin(), weights.end());

  // new particles
  std::vector<Particle> new_particles(particles);

  for (size_t i = 0; i < particles.size(); ++i) {
    const int index = d(gen);
    new_particles.push_back(particles[index]);
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