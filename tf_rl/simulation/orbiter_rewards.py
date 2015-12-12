import math
import numpy as np

class Orbit(object):
    def __init__(self, shape = 'circle', center = np.array([0, 0]), sigma = 50, maximum = 10, radius = 1, radius2 = None):
        self.shape = shape
        self.sigma = sigma
        self.maximum = maximum
        self.center = center
        self.radius = radius
        self.radius2 = radius2

    def distance(self, position):
        if self.shape == 'circle':
            return abs(np.sqrt((position - self.center).dot(position - self.center)) - self.radius)
        elif self.shape == 'ellipse':
            return 0.1

    def reward(self, position):
        x = self.distance(position)
        prob = np.exp(-(x**2)/(2 * self.sigma**2))
        return -(prob * self.maximum) + 10
		
