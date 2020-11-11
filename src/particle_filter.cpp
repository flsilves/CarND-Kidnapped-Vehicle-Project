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
constexpr auto number_of_particles{200};  // make me

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Add noise to initial position using GPS standard deviation
  const auto init_x = dist_x(gen);
  const auto init_y = dist_y(gen);
  const auto init_theta = dist_theta(gen);

  constexpr auto init_weight{1.0};

  particles.reserve(number_of_particles);

  for (int id = 1; id <= number_of_particles; ++id) {
    particles.emplace_back(
        Particle{id, init_x, init_y, init_theta, init_weight});
  }

  is_initialized = true;

  // std::cout << "associations:" << particles[0].associations.size() <<
  // std::endl; std::cout << "x:" << particles[0].sense_x.size() << std::endl;
  // std::cout << "y:" << particles[0].sense_y.size() << std::endl;
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
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no
   * scaling). The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
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