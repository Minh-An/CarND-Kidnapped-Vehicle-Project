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

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    num_particles = 500; // TODO: Set the number of particles
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    particles = vector<Particle>(num_particles);
    for (int i = 0; i < num_particles; ++i)
    {
        particles[i] = Particle(i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0);
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
    /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);
    for(Particle& p: particles) {
        double theta_f = p.theta + yaw_rate*delta_t;
        double x_f = p.x + velocity/yaw_rate * (sin(theta_f) - sin(p.theta));
        double y_f = p.y + velocity/yaw_rate * (cos(p.theta) - cos(theta_f));
        p.theta = theta_f + dist_theta(gen);
        p.x = x_f + dist_x(gen);
        p.y = y_f + dist_y(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
    /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for(LandmarkObs& obs: observations) {
        double minDist = std::numeric_limits<double>::infinity();
        for(LandmarkObs p: predicted) {
            double distance = dist(obs.x, obs.y, p.x, p.y);
            if(distance < minDist) {
                minDist = distance;
                obs.id = p.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
    /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    for(Particle& p: particles) {
        vector<LandmarkObs> transformedObservations(observations.size());
        std::transform(observations.begin(), observations.end(), transformedObservations.begin(), 
            [p] (const LandmarkObs& obs) {
                LandmarkObs transformedOb;
                transformedOb.x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
                transformedOb.y = p.y + (sin(p.theta) *  obs.x) + (cos(p.theta) * obs.y);
                return transformedOb;
            });
        vector<LandmarkObs> predictions;
        for(Map::single_landmark_s landmark: map_landmarks.landmark_list) {
            if(dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range) {
                LandmarkObs pred;
                pred.x = landmark.x_f;
                pred.y = landmark.y_f;
                pred.id = landmark.id_i;
                predictions.push_back(pred);
            }
        }
        dataAssociation(predictions, transformedObservations);
        double finalWeight = 1;
        for(LandmarkObs obs: transformedObservations) {
            auto it = std::find_if(predictions.begin(), predictions.end(), [obs] (const LandmarkObs& landmark) {return landmark.id == obs.id;});
            double coeff = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
            double x_diff_sq = pow(obs.x - it->x, 2);
            double y_diff_sq = pow(obs.y - it->y, 2);
            double x_denom = 2*pow(std_landmark[0], 2);
            double y_denom = 2*pow(std_landmark[1], 2);
            double exponent = exp(-(x_diff_sq/x_denom + y_diff_sq/y_denom));
            double prob = coeff*exponent;
            finalWeight *= prob;
        }
        p.weight = finalWeight;
    }
}

void ParticleFilter::resample()
{
    /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    vector<double> weights(particles.size());
    std::transform(particles.begin(), particles.end(), weights.begin(), [] (const Particle& p) { return p.weight;});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());

    vector<Particle> newParticles(particles.size());
    for(int i = 0; i < newParticles.size(); i++) {
        newParticles[i] = particles[d(gen)];
    }
    particles = newParticles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
    vector<double> v;

    if (coord == "X")
    {
        v = best.sense_x;
    }
    else
    {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}