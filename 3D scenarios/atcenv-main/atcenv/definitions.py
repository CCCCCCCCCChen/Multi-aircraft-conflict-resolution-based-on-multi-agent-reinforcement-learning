"""
Definitions module
"""
from shapely.geometry import Point, Polygon
from dataclasses import dataclass, field
import atcenv.units as u
import math
import random
from typing import Optional, Tuple


@dataclass
class Airspace:
    """
    Airspace class
    """
    polygon: Polygon

    @classmethod
    def random(cls, min_area: float, max_area: float):
        """
        Creates a random airspace sector with min_area < area <= max_area

        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :return: random airspace
        """
        R = math.sqrt(max_area / math.pi)

        def random_point_in_circle(radius: float) -> Point:
            alpha = 2 * math.pi * random.uniform(0., 1.)
            r = radius * math.sqrt(random.uniform(0., 1.))
            x = r * math.cos(alpha)
            y = r * math.sin(alpha)
            return Point(x, y)

        p = [random_point_in_circle(R) for _ in range(3)]
        polygon = Polygon(p).convex_hull

        while polygon.area < min_area:
            p.append(random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        return cls(polygon=polygon)


@dataclass
class Flight:
    """
    Flight class
    """
    position: Point
    target: Point
    optimal_airspeed: float
    optimal_level: float  # 0: FL300, 1: FL310, 2: FL320, 3: FL330, [4: FL340, 5: FL350]
    max_level: int
    min_level: int

    airspeed: float = field(init=False)
    track: float = field(init=False)
    level: float = field(init=False)
    optimal_trajectory_length: float = field(init=False)

    action: int = field(init=False)
    intention_airspeed: int = field(init=False)
    intention_heading: int = field(init=False)
    intention_level: int = field(init=False)
    last_intention_airspeed: int = 0
    last_intention_heading: int = 0
    last_level: float = field(init=False)

    closest_distance: int = 99999999
    trajectory_length: float = 0
    r_conflict: int = 0
    r_warning: int = 0
    r_eco_airspeed: int = 0
    r_eco_heading: int = 0
    r_eco_level: int = 0
    r_smo_airspeed: int = 0
    r_smo_heading: int = 0
    r_smo_level: int = 0
    r_over_max_level: int = 0
    r_over_min_level: int = 0

    def __post_init__(self) -> None:
        """
        Initialises the track and the airspeed
        :return:
        """
        self.track = self.bearing
        self.airspeed = self.optimal_airspeed
        self.level = self.optimal_level
        self.last_level = self.optimal_level
        self.optimal_trajectory_length = self.position.distance(self.target)

    @property
    def bearing(self) -> float:
        """
        Bearing from current position to target
        :return:
        """
        dx = self.target.x - self.position.x
        dy = self.target.y - self.position.y
        compass = math.atan2(dy, dx)
        return (compass + u.circle) % u.circle

    @property
    def prediction(self, dt: Optional[float] = 120) -> Point:
        """
        Predicts the future position after dt seconds, maintaining the current speed and track
        :param dt: prediction look-ahead time (in seconds)
        :return:
        """
        dx, dy = self.components
        return Point([self.position.x + dx * dt, self.position.y + dy * dt])

    @property
    def components(self) -> Tuple:
        """
        X and Y Speed components (in kt)
        :return: speed components
        """
        dx = self.airspeed * math.cos(self.track)
        dy = self.airspeed * math.sin(self.track)
        return dx, dy

    @property
    def distance(self) -> float:
        """
        Current distance to the target (in meters)
        :return: distance to the target
        """
        return self.position.distance(self.target)

    @property
    def drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target
        :return:
        """
        drift = self.bearing - self.track
        if -math.pi < drift <= math.pi:
            return drift
        elif drift <= -math.pi:
            return drift + u.circle
        elif drift > math.pi:
            return drift - u.circle

    @classmethod
    def random(cls, airspace: Airspace, min_speed: float, max_speed: float, tol: float = 0., num_levels: int = 4):
        """
        Creates a random flight

        :param airspace: airspace where the flight is located
        :param min_speed: minimum speed of the flights (in kt)
        :param max_speed: maximum speed of the flights (in kt)
        :param tol: tolerance to consider that the target has been reached (in meters)
        :param num_levels: number of flight levels

        :return: random flight
        """
        def random_point_in_polygon(polygon: Polygon) -> Point:
            minx, miny, maxx, maxy = polygon.bounds
            while True:
                point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if polygon.contains(point):
                    return point

        # random position
        position = random_point_in_polygon(airspace.polygon)

        # random target
        boundary = airspace.polygon.boundary
        while True:
            d = random.uniform(0, airspace.polygon.boundary.length)
            target = boundary.interpolate(d)
            if target.distance(position) > tol:
                break

        # random speed
        airspeed = random.uniform(min_speed, max_speed)

        # random flight level
        max_level = num_levels - 1
        min_level = 0
        level = random.randint(min_level, max_level)

        return cls(position=position, target=target, optimal_airspeed=airspeed, optimal_level=level, max_level=max_level, min_level=min_level)
