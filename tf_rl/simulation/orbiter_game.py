import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import logging

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2

from orbiter_rewards import Orbit

import tf_rl.utils.svg as svg

class GameObject(object):
    def __init__(self, position, speed, radius, mass, heading, obj_type, settings):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.settings = settings
        self.radius = radius
        self.mass = mass

        self.obj_type = obj_type
        self.position = position
        self.speed    = speed
        self.heading  = heading

    def step(self, dt):
        """Move as if dt seconds passed"""
        self.position += dt * self.speed

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self, scale):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        if self.obj_type == "craft":
            return svg.Circle(self.position*scale + Point2(10, 10), 3, color=color)
        return svg.Circle(self.position*scale + Point2(10, 10), self.radius*scale, color=color)

class OrbiterGame(object):
    def __init__(self, settings):
        """Initiallize game simulator with settings"""
        self.logfile = open('./log.txt', 'w+')
        self.logfile.write('Altitude,Reward,Orbit Angle,Start\n');
        self.settings = settings
        self.size  = self.settings["world_size"]
        self.sim_time = 0.

        self.G = self.settings["G"]
        self.gravity = np.array([0, 0])

        self.reset = False
        self.init_craft_planet()
        self.fuel_cost = -1 / self.settings["craft_max_thrust"]

        # objects == asteroids
        self.asteroid_mass = self.settings["asteroid_mass"]
        self.asteroid_radius = self.settings["asteroid_radius"]
        self.objects = []
        for obj_type, number in settings["num_objects"].items():
            for _ in range(number):
                self.spawn_object(obj_type)

        self.observation_lines = self.generate_observation_lines()

        self.object_reward = 0
        self.collected_rewards = []

        # every observation_line sees one of objects or wall and
        # two numbers representing speed of the object (if applicable)
        self.eye_observation_size = 1 #len(self.settings["objects"]) + 2 # 2 not 3 because we have no walls
        # additionally there are two numbers representing agents own speed.
        self.observation_size = self.eye_observation_size * len(self.observation_lines) + 4

        rotations = self.settings["craft_rotations"]
        thrusts = range(self.settings["craft_min_thrust"],
                        self.settings["craft_max_thrust"],
                        self.settings["craft_step_thrust"])

        self.directions = list(itertools.product(rotations, thrusts))
        self.num_actions = len(self.directions)

        self.objects_eaten = defaultdict(lambda: 0)

    def linear_reward(self, position):
#        if np.linalg.norm(position - self.planet.position) < self.orbit.radius:
         return -(np.linalg.norm(self.orbit.radius + self.planet.position - position)/self.orbit.radius)**2
#        else:
#            return -np.linalg.norm(self.orbit.radius + self.planet.position - position)/self.orbit.radius

    def init_craft_planet(self):
        self.craft = GameObject(np.array(self.settings["craft_initial_position"], dtype=float),
            np.array(self.settings["craft_initial_speed"], dtype=float),
            np.array(self.settings["craft_radius"], dtype=float),
            np.array(self.settings["craft_mass"], dtype=float),
            np.array(self.settings["craft_thrust_angle"], dtype=float),
            "craft", 
            self.settings)


        self.planet = GameObject(np.array(self.settings["planet_initial_position"], dtype=float),
            np.array(self.settings["planet_initial_speed"], dtype=float),
            np.array(self.settings["planet_radius"], dtype=float), 
            np.array(self.settings["planet_mass"], dtype=float),
            0,
            "planet",
            self.settings)

        self.orbit = Orbit(center=self.planet.position, 
            radius=self.planet.radius+self.settings["orbit_altitude"])


    def perform_action(self, action_id):
        """Change speed to one of craft vectors"""
        assert 0 <= action_id < self.num_actions

        self.craft.heading += self.directions[action_id][0]

        thrust_vec = -np.array([np.cos(self.craft.heading),
                                np.sin(self.craft.heading)])

        self.craft.speed += self.directions[action_id][1] * thrust_vec

        self.object_reward += self.directions[action_id][1] * self.fuel_cost
        self.object_reward += abs(self.directions[action_id][0])*(-.025)

    def spawn_object(self, obj_type):
        # TODO: avoid placement too close to craft / inside planet
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["asteroid_radius"]
        position = np.random.uniform([radius, radius], np.array(self.size) - radius)
        position = np.array([float(position[0]), float(position[1])])
        max_speed = np.array(self.settings["maximum_speed"])
        speed = np.random.uniform(-max_speed, max_speed, 2).astype(float)

        self.objects.append(GameObject(position, speed, 
                            self.asteroid_radius, self.asteroid_mass,
                            obj_type, self.settings))

    def step(self, dt):
        """Simulate all the objects for a given ammount of time.

        Also resolve collisions with the craft"""
        for obj in self.objects + [self.craft] :
            r = self.planet.position - obj.position
            g = (self.G * self.planet.mass * obj.mass) \
                / (np.linalg.norm(r) ** 2)

            force = g * (r / np.linalg.norm(r))

            if obj == self.craft:
                self.gravity = force

            obj.speed += dt * force / obj.mass
            obj.step(dt)

        self.resolve_collisions()

        unitThrust = np.array([math.cos(self.craft.heading), math.sin(self.craft.heading)])
        pToC = self.craft.position - self.planet.position
        cToO = np.array([-1*pToC[1], pToC[0]])
        orbitAngle = np.arccos(unitThrust.dot(cToO)/(np.linalg.norm(cToO)*np.linalg.norm(unitThrust)))
        altitude = (np.linalg.norm(self.craft.position - self.planet.position) - self.planet.radius)

        self.logfile.write("%.1f,%.3f,%.3f," % (altitude, (sum(self.collected_rewards[-1:])), orbitAngle))

        if self.reset:
            self.object_reward -= 10
            self.reset = False
            self.logfile.write('1\n')
#            print('Resetting')
        else:
            # self.object_reward += self.orbit.reward(self.craft.position)
            self.logfile.write('0\n')
            self.object_reward += self.linear_reward(self.craft.position) 

        boundary = self.planet.radius +self.settings["orbit_altitude"]

        if altitude > boundary:
            self.init_craft_planet()
            self.reset = True

        self.sim_time += dt

    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def resolve_collisions(self):
        """If craft touches, reward gets updated."""
        to_remove = []
        for obj in self.objects + [self.planet] :
            collision_distance2 = (self.craft.radius + obj.radius) ** 2
            if self.squared_distance(self.craft.position, obj.position) < collision_distance2:
                to_remove.append(obj)

        for obj in to_remove:
            self.objects_eaten[obj.obj_type] += 1
            
            if obj.obj_type == "planet":
                self.init_craft_planet()
                self.reset = True
                self.object_reward += self.linear_reward(self.size[0])
            else:
                self.objects.remove(obj)
                self.spawn_object(obj.obj_type)
                self.object_reward += self.settings["object_reward"][obj.obj_type]
                

    def observe(self):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        
        self.orbit = Orbit(center=self.planet.position, 
                            radius=self.planet.radius+self.settings["orbit_altitude"])
        pos = Point2(*(self.craft.position).tolist())
        self.observation_lines = self.generate_observation_lines()
        num_obj_types = len(self.settings["objects"]) # no plus one, we don't have walls
        max_speed_x, max_speed_y = self.settings["maximum_speed"]

        observable_distance = self.settings["observation_line_length"]

#        # Filters out asteroids outside of observations lines
#        relevant_asteroids = [obj for obj in self.objects
#                            if obj.position.distance(pos) < observable_distance]
#
#        # objects sorted from closest to furthest
#        relevant_asteroids.sort(key=lambda x: x.position.distance(pos))
#
        observation        = np.zeros(self.observation_size)
        observation_offset = 0
#        for i, observation_line in enumerate(self.observation_lines):
#            # shift to craft position
#
#            start = pos + Vector2(*observation_line.p1)
#            end = pos + Vector2(*observation_line.p2)
#            observation_line = LineSegment2(start, end)
#
#            observed_object = Noneltitude(m)
#            for obj in relevant_asteroids:
#                if observation_line.distance(obj.position) < obj.radius:
#                    observed_object = obj
#                    break
#            object_type_id = None
#            speed_x, speed_y = 0, 0
#            proximity = 0
#            if observed_object is not None: # agent seen
#                object_type_id = self.settings["objects"].index(observed_object.obj_type)
#                speed_x, speed_y = tuple(observed_object.speed)
#                intersection_segment = obj.as_circle().intersect(observation_line)
#                assert intersection_segment is not None
#                try:
#                    proximity = min(intersection_segment.p1.distance(pos),
#                                    intersection_segment.p2.distance(pos))
#                except AttributeError:
#                    proximity = observable_distance
#            else:
#                accuracy = 10.
#                rewards = np.array([])
#                for i in range(1, int(accuracy+1)):
#                    coordinates = np.empty(2, dtype=float) 
#                    coordinates[0] = i*(end.x - start.x)/accuracy + start.x
#                    coordinates[1] = i*(end.y - start.y)/accuracy + start.y
#                    # rewards = np.append(rewards, self.orbit.reward(coordinates))
#                    rewards = np.append(rewards, self.linear_reward(self.craft.position)) 
#            observation[observation_offset] = np.around(np.average(rewards), decimals=1)
#            for object_type_idx_loop in range(num_obj_types):
#                observation[observation_offset + object_type_idx_loop] = 1.0
#            if object_type_id is not None:
#                observation[observation_offset + object_type_id] = proximity / observable_distance
#            observation[observation_offset + num_obj_types] =     speed_x   / max_speed_x
#            observation[observation_offset + num_obj_types + 1] = speed_y   / max_speed_y
            #assert num_obj_types + 2 == self.eye_observation_size
#           observation_offset += self.eye_observation_size

        unitThrust = np.array([math.cos(self.craft.heading), math.sin(self.craft.heading)])
        pToC = self.craft.position - self.planet.position
        cToO = np.array([-1*pToC[1], pToC[0]])
        orbitAngle = np.arccos(unitThrust.dot(cToO)/(np.linalg.norm(cToO)*np.linalg.norm(unitThrust)))

        observation[observation_offset]     = np.linalg.norm(self.craft.speed.dot(unitThrust)*unitThrust)
        observation[observation_offset + 1] = np.linalg.norm(self.craft.speed - observation[observation_offset])
        observation[observation_offset + 2] = (np.linalg.norm(self.craft.position - self.planet.position) - self.planet.radius)
        observation[observation_offset + 3] = orbitAngle
    
 
#        observation[observation_offset + 2] = self.gravity[0]
#        observation[observation_offset + 3] = self.gravity[1]
#        assert observation_offset + 4 == self.observation_size
		
        return observation

    def collect_reward(self):
        """Return accumulated object eating score + current distance to walls score"""
        total_reward = self.object_reward
        self.object_reward = 0
        self.collected_rewards.append(total_reward)
        return total_reward

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []
        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                        self.settings["observation_line_length"])
        thrustAngle = np.radians(self.craft.heading)
        for angle in np.linspace(thrustAngle, thrustAngle, self.settings["num_observation_lines"], endpoint=True):
            rotation = Point2(math.cos(angle), -math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        scale = self.settings["image_size"] / self.settings["world_size"][0]

#        unitThrust = np.array([math.cos(self.craft.heading), math.sin(self.craft.heading)])
#        pToC = self.craft.position - self.planet.position
#        cToO = np.array([-1*pToC[1], pToC[0]])
#        orbitAngle = np.arccos(unitThrust.dot(cToO)/(np.linalg.norm(cToO)*np.linalg.norm(unitThrust)))

        stats = stats[:]
        reward = self.collected_rewards + [0]
        recent_reward = reward[-101:]
        altitude = (np.linalg.norm(self.craft.position - self.planet.position) - self.planet.radius)
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            "time         = %.1f s" % (self.sim_time),
        "altitude     = %.1f m" % (altitude),
            "gravity      = %.1f N" % (np.linalg.norm(self.gravity)),
            "speed        = %.1f m/s" % (np.linalg.norm(self.craft.speed)),
            "heading      = %.1f degrees" % (self.craft.heading),
            "last_reward  = %.3f" % (sum(self.collected_rewards[-1:])),
            "recent_reward = %.3f" % (sum(recent_reward)/len(recent_reward)),
            "total_reward = %.3f" % (sum(reward)/len(reward)),
        ])

#        self.logfile.write("%.1f,%.3f,%.3f," % (altitude, (sum(self.collected_rewards[-1:])), orbitAngle))
        
        scene = svg.Scene((self.size[0] * scale + 20,
                            self.size[1] * scale + 20 + 20 * len(stats)))

        scene.add(svg.Circle(self.planet.position*scale+Point2(10,10), \
            (self.planet.radius+2*self.settings["orbit_altitude"])*scale, \
            color='white', stroke='red'))
        scene.add(svg.Circle(self.planet.position*scale+Point2(10,10), \
            (self.planet.radius+self.settings["orbit_altitude"])*scale, \
            color='white', stroke='green'))
        scene.add(svg.Rectangle((10, 10), [x * scale for x in self.size]))
        scene.add(self.planet.draw(scale))


        for line in self.observation_lines:
            scene.add(svg.Line(line.p1 + Point2(*(self.craft.position).tolist()) * scale + Point2(10,10),
                               line.p2 + Point2(*(self.craft.position).tolist()) * scale + Point2(10,10)))

        for obj in self.objects + [self.craft] :
            scene.add(obj.draw(scale))

        offset = self.size[1]*scale + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene
