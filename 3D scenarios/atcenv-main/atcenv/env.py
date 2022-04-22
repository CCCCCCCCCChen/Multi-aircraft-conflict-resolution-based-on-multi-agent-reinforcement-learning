"""
Environment module
"""
import gym
from typing import List
import numpy as np
from atcenv.definitions import *
from gym.envs.classic_control import rendering
from shapely.geometry import LineString
from scipy.stats import burr


"""
Colour options for rendering if needed
Note: the range of rgb in rendering is [0, 1] rather than [0, 255]
"""
WHITE = [1, 1, 1]
RED = [1, 0, 0]
ORANGE = [1, 0.5, 0]
YELLOW = [0.8, 1, 0.1]
GREEN = [0, 1, 0]
BLUE = [0.2, 0.3, 1]
CYAN = [0, 1, 1]
PURPLE = [1, 0, 1]
BLACK = [0, 0, 0]


def calculate_angel(a: Point, b: Point):
    """calculate the relative angle, range [0, 2*pi)"""
    dx = b.x - a.x
    dy = b.y - a.y
    relative_angle = math.atan2(dy, dx)
    relative_angle = (relative_angle + 2 * math.pi) % (2 * math.pi)
    return relative_angle


def angel_clip(angel):
    """clip the angle in [-pi, pi)"""
    while angel < -math.pi or angel >= math.pi:
        if angel < -math.pi:
            angel += (2 * math.pi)
        if angel >= math.pi:
            angel -= (2 * math.pi)
    return angel


class Environment(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 max_area: Optional[float] = 250. * 250.,  # nm^2
                 min_area: Optional[float] = 125. * 125.,  # nm^2
                 num_levels: int = 4,
                 num_flights: int = 60,
                 distance_init_buffer: Optional[float] = 5.,
                 min_distance: Optional[float] = 4.,  # nm
                 max_speed: Optional[float] = 470.,  # kt
                 min_speed: Optional[float] = 430,  # kt
                 heading_change: Optional[float] = math.pi / 3,  # rad
                 airspeed_change: Optional[float] = 30,  # kt
                 dt: float = 6.,  # second
                 num_move: Optional[int] = 5,
                 max_episode_len: Optional[int] = 500,
                 num_detection_sectors: Optional[int] = 12,
                 range_detection: Optional[float] = 30.,  # nm
                 heading_error_scale: Optional[float] = math.pi / 450,  # rad
                 airspeed_error_scale: Optional[float] = 2.5,  # percentage
                 max_wind_speed: Optional[float] = 50,  # m/s
                 wind_speed_range: Optional[float] = 5,  # m/s
                 ):
        """
        Initialises the environment

        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :param num_levels: number of flight levels
        :param num_flights: number of flights in the environment
        :param distance_init_buffer: distance factor used when initialising the environment
        to avoid flights close to conflict and close to the target
        :param min_distance: pairs of flights which distance is < min_distance are considered in conflict (in nm)
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param heading_change: Limit of change of heading per action (in rad)
        :param airspeed_change: difference between the max/min speed and optimal speed (in kt)
        :param dt: time step (in seconds)
        :param num_move: number of movements (time steps) for an action
        :param max_episode_len: maximum episode length (in number of steps)
        :param num_detection_sectors: number of sectors in which the detection area is divided
        :param range_detection: radius of the detection area (in nm)
        :param heading_error_scale: scale of heading error (in rad)
        :param airspeed_error_scale: scale of airspeed error (in percentage)
        :param max_wind_speed: the maximum speed of wind (in m/s)
        :param wind_speed_range: the range of wind speed based on the reference wind in a scenario (in m/s)
        """
        self.num_flights = num_flights
        self.max_area = max_area * (u.nm ** 2)
        self.min_area = min_area * (u.nm ** 2)
        self.max_speed = max_speed * u.kt
        self.min_speed = min_speed * u.kt
        self.min_distance = min_distance * u.nm
        self.max_episode_len = max_episode_len
        self.distance_init_buffer = distance_init_buffer
        self.dt = dt
        self.num_move = num_move
        self.heading_change = heading_change
        self.airspeed_change = airspeed_change * u.kt
        self.num_levels = num_levels
        self.num_detection_sectors = num_detection_sectors
        self.range_detection = range_detection * u.nm
        self.range_warning = min_distance * 2 * u.nm  # radius of the warning area
        self.range_protection = min_distance * u.nm  # radius of the protection area

        # tolerance to consider that the target has been reached (in meters)
        self.tol = self.max_speed * 1.05 * self.dt

        # uncertainty
        self.heading_error_scale = heading_error_scale
        self.airspeed_error_scale = airspeed_error_scale

        # weather
        self.max_wind_speed = max_wind_speed
        self.wind_speed_range = wind_speed_range
        self.reference_wind_heading = random.uniform(0, 1) * 2 * math.pi
        self.reference_wind_speed = np.clip(burr.rvs(c=4.089, d=0.814, loc=-0.042, size=1, scale=17.47),
                                            self.wind_speed_range, self.max_wind_speed - self.wind_speed_range)

        self.viewer = None
        self.airspace = None
        self.flights = []  # list of flights
        self.conflicts = set()  # set of flights that are in conflict in a step
        self.warning = set()  # set of warning that are in conflict in a step
        self.period_conflicts = set()  # set of flights that are in conflict in an action cycle
        self.period_warning = set()  # set of flights that are in warning in an action cycle
        self.done = set()  # set of flights that reached the target
        self.i = None  # time step

        self.intention_matrix = np.array([_ for _ in range(27)]).reshape([3, 3, 3])
        # the intention matrix is used to decompose the action into heading intention and speed intention

    def resolution(self, action: List, t) -> None:
        """
        Applies the resolution actions
        :param action: list of resolution actions assigned to each flight
        :param t: t/num_move -th step of an action cycle
        :return: None
        """

        for i in range(self.num_flights):
            if i not in self.done:
                # decompose the action into heading intention and speed intention
                # heading intention:
                # -1 means flying to the target from right
                # 0 means flying straight to the target
                # 1 means flying to the target from left
                # speed intention:
                # -1 means flying with the low speed
                # 0 means flying with the optimal speed
                # 1 means flying with the high speed
                # level intention:
                # -1 means descending to the next flight level
                # 0 means maintaining flight at the current flight level
                # 1 means climbing to the next flight level
                self.flights[i].intention_airspeed, self.flights[i].intention_heading, self.flights[i].intention_level \
                    = np.where(self.intention_matrix == action[i])
                self.flights[i].intention_airspeed -= 1
                self.flights[i].intention_heading -= 1
                self.flights[i].intention_level -= 1

                # heading supervisor
                if abs(self.flights[i].last_intention_heading - self.flights[i].intention_heading) == 2:
                    self.flights[i].intention_heading = 0

                # airspeed change
                airspeed_error = np.clip(random.normalvariate(0, 1), -1, 1) * self.airspeed_error_scale / 100
                self.flights[i].airspeed = self.flights[i].optimal_airspeed + self.airspeed_change * self.flights[
                        i].last_intention_airspeed + self.airspeed_change * (t + 1) / self.num_move * (
                        self.flights[i].intention_airspeed - self.flights[
                            i].last_intention_airspeed) * (1 + airspeed_error)

                # heading change
                heading_error = np.clip(random.normalvariate(0, 1), -1, 1) * self.heading_error_scale
                self.flights[i].track = \
                    self.flights[i].bearing + self.heading_change * self.flights[i].last_intention_heading + \
                    self.heading_change * (t + 1) / self.num_move * (self.flights[i].intention_heading - self.flights[
                        i].last_intention_heading) + heading_error

                # level change
                if self.flights[i].intention_level == -1:
                    if self.flights[i].level > self.flights[i].min_level:
                        self.flights[i].level = round(self.flights[i].level - 1/self.num_move, 1)
                elif self.flights[i].intention_level == 0:
                    pass
                elif self.flights[i].intention_level == 1:
                    if self.flights[i].level < self.flights[i].max_level:
                        self.flights[i].level = round(self.flights[i].level + 1/self.num_move, 1)
        return None

    def reward(self) -> List:
        """
        Returns the reward assigned to each agent
        :return: reward assigned to each agent
        the reward considers: conflict, warning, economy [, smoothness], legality
        """

        reward = []
        for i in range(self.num_flights):
            if i in self.done:
                reward.append(np.nan)
            else:
                # conflict
                self.flights[i].r_conflict = -1 if i in self.period_conflicts else 0
                # warning
                self.flights[i].r_warning = -1 if i in self.period_warning else 0
                # economy
                self.flights[i].r_eco_airspeed = -1 if self.flights[i].intention_airspeed != 0 else 0
                self.flights[i].r_eco_heading = -1 if self.flights[i].intention_heading != 0 else 0
                self.flights[i].r_eco_level = -abs(self.flights[i].level - self.flights[i].optimal_level)
                # smoothness
                self.flights[i].r_smo_airspeed = -1 if self.flights[i].intention_airspeed != self.flights[
                    i].last_intention_airspeed else 0
                self.flights[i].r_smo_heading = -1 if self.flights[i].intention_heading != self.flights[
                    i].last_intention_heading else 0
                self.flights[i].r_smo_level = -1 if self.flights[i].intention_level != 0 else 0
                # legality
                self.flights[i].r_over_max_level = -1 if self.flights[i].last_level == self.flights[i].max_level and \
                    self.flights[i].intention_level == 1 else 0
                self.flights[i].r_over_min_level = -1 if self.flights[i].last_level == self.flights[i].min_level and \
                    self.flights[i].intention_level == -1 else 0

                # Proportion of invasion of the protection area at distance
                p_conflict = max((self.range_protection - self.flights[i].closest_distance) / self.range_protection, 0)
                # Proportion of invasion of the warning area at distance
                p_warning = max((self.range_warning - self.flights[i].closest_distance) / self.range_warning, 0)
                # reward
                r = 200 * (1 + p_conflict) * self.flights[i].r_conflict \
                    + 20 * (1 + p_warning) * self.flights[i].r_warning \
                    + 2 * self.flights[i].r_eco_airspeed + 5 * self.flights[i].r_eco_heading + 2 * self.flights[
                        i].r_eco_level \
                    + 1 * self.flights[i].r_smo_airspeed + 1 * self.flights[i].r_smo_heading + 10 * self.flights[
                        i].r_smo_level \
                    + 50 * (self.flights[i].r_over_max_level + self.flights[i].r_over_min_level)
                reward.append(r)
        return reward
        ##########################################################

    def observation(self) -> List:
        """
        Returns the observation of each agent
        :return: observation of each agent

        owner info:
            aircraft type: represented by optimal_speed, maximum flight level, minimum flight level
            flight status: last_intention_heading, last_intention_speed
            intention: distance to target, optimal flight level

        intruder info (a sector of detection area):
            * using polar coordinates (origin: owner's position, x-axis: owner's bearing)
            occupation label
            aircraft type: represented by optimal_speed
            position: distance (intruder -- owner), angle (intruder -> owner)
            flight status: track, speed
            intention: bearing, distance to target, optimal flight level
        """

        observation = []
        for i in range(self.num_flights):
            if i in self.done:
                observation.append(np.nan)
            else:
                # intruder's information
                intruder_info = np.zeros([3, self.num_detection_sectors, 3])
                for j in range(self.num_flights):
                    if j != i and j not in self.done:
                        distance_i_j = self.flights[i].position.distance(self.flights[j].position)
                        level_i_j = self.flights[i].level - self.flights[j].level
                        if distance_i_j <= self.range_detection and abs(level_i_j) <= 1:
                            # intruder info:
                            level_idx = int(level_i_j + 1)
                            position_angle_i_j = angel_clip(
                                calculate_angel(self.flights[i].position, self.flights[j].position) - self.flights[
                                    i].bearing)
                            sector_idx = int(position_angle_i_j // (
                                        2 * math.pi / self.num_detection_sectors) + self.num_detection_sectors / 2)
                            if sector_idx == self.num_detection_sectors:
                                sector_idx -= 1
                            intrusion_distance = self.range_detection - distance_i_j
                            if intrusion_distance > intruder_info[level_idx, sector_idx, 1]:
                                intruder_info[level_idx, sector_idx, 0] = 1.  # label, if being occupied
                                intruder_info[level_idx, sector_idx, 1] = intrusion_distance  # distance
                                intruder_info[level_idx, sector_idx, 2] = angel_clip(
                                    self.flights[j].bearing - self.flights[i].bearing)  # bearing
                # owner's information
                owner_info = np.array([self.flights[i].level - self.flights[i].optimal_level,  # level change
                                       self.flights[i].level - self.flights[i].min_level,  # minimum
                                       self.flights[i].max_level - self.flights[i].level,  # maximum
                                       ])
                ob = np.concatenate((intruder_info.flatten(), owner_info), axis=0)
                observation.append(ob)

        return observation

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        Note: flights that reached the target are not considered
        :return: None
        """
        # reset set
        self.conflicts = set()
        self.warning = set()

        for i in range(self.num_flights - 1):
            if i not in self.done:
                for j in range(i + 1, self.num_flights):
                    if (j not in self.done) and abs(self.flights[i].level - self.flights[j].level) < 1:
                        distance = self.flights[i].position.distance(self.flights[j].position)
                        if distance < self.range_warning:
                            self.warning.update((i, j))
                            self.period_warning.update((i, j))
                        if distance < self.range_protection:
                            self.conflicts.update((i, j))
                            self.period_conflicts.update((i, j))
                        # update the closest distance in the action cycle
                        if distance < self.flights[i].closest_distance:
                            self.flights[i].closest_distance = distance
                        if distance < self.flights[j].closest_distance:
                            self.flights[j].closest_distance = distance

    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        :return: None
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                distance = f.position.distance(f.target)
                if distance < self.tol:
                    self.done.add(i)

    def update_positions(self) -> None:
        """
        Updates the position of the agents
        Note: the position of agents that reached the target is not modified
        :return: None
        """
        # get wind speed
        wind_heading = self.reference_wind_heading
        wind_speed = self.reference_wind_speed + np.clip(random.normalvariate(0, 1), -1, 1) * self.wind_speed_range
        wind_dx = math.cos(wind_heading) * wind_speed
        wind_dy = math.sin(wind_heading) * wind_speed

        for i, f in enumerate(self.flights):
            if i not in self.done:
                # get current speed components
                dx, dy = f.components
                # get current position
                position = f.position
                # get new position and advance one time step
                f.position._set_coords(position.x + (dx + wind_dx) * self.dt, position.y + (dy + wind_dy) * self.dt)
                # compute the length of the trajectory so far
                f.trajectory_length += (((dx + wind_dx) * self.dt) ** 2 + ((dy + wind_dy) * self.dt) ** 2) ** 0.5

    def step(self, action: List) -> Tuple[List, List, bool]:
        """
        Performs an action simulation with num_move steps

        :param action: list of resolution actions assigned to each flight
        :return: observation, reward, and done status
        """

        for t in range(self.num_move):
            # increase steps counter
            self.i += 1

            # apply resolution actions
            self.resolution(action, t)

            # update positions
            self.update_positions()

            # update done set
            self.update_done()

            # update conflict set
            self.update_conflicts()

            # render
            # self.render()

        # compute reward
        rew = self.reward()

        # compute observation
        obs = self.observation()

        # record the last intentions
        for i in range(self.num_flights):
            self.flights[i].last_intention_airspeed = self.flights[i].intention_airspeed
            self.flights[i].last_intention_heading = self.flights[i].intention_heading
            self.flights[i].last_level = self.flights[i].level

        # check termination status
        # termination happens when
        # (1) all flights reached the target
        # (2) the maximum episode length is reached
        done = (len(self.done) == self.num_flights or self.i > self.max_episode_len)

        # clean the conflict and warning set
        self.period_conflicts = set()
        self.period_warning = set()

        # reset the closest distance
        for i in range(self.num_flights):
            self.flights[i].closest_distance = 99999999

        return rew, obs, done

    def reset(self) -> List:
        """
        Resets the environment and returns initial observation
        :return: initial observation
        """
        # create random airspace
        self.airspace = Airspace.random(self.min_area, self.max_area)

        # create random flights
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance
        while len(self.flights) < self.num_flights:
            valid = True
            candidate = Flight.random(self.airspace, self.min_speed, self.max_speed, tol, self.num_levels)

            # ensure that candidate is not in conflict
            for f in self.flights:
                if (candidate.position.distance(f.position) < min_distance) and (f.level == candidate.level):
                    valid = False
                    break
            if valid:
                self.flights.append(candidate)

        # initialise steps counter
        self.i = 0

        # clean conflicts, warning and done sets
        self.conflicts = set()
        self.warning = set()
        self.period_conflicts = set()
        self.period_warning = set()
        self.done = set()

        # create reference wind
        self.reference_wind_heading = random.uniform(0, 1) * 2 * math.pi
        self.reference_wind_speed = np.clip(burr.rvs(c=4.089, d=0.814, loc=-0.042, size=1, scale=17.47),
                                            self.wind_speed_range, self.max_wind_speed - self.wind_speed_range)

        # return initial observation
        return self.observation()

    def render(self, mode=None) -> None:
        """
        Renders the environment
        :param mode: rendering mode
        :return: None
        """
        if self.viewer is None:
            # initialise viewer
            screen_width, screen_height = 1200, 1200

            minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(minx, maxx, miny, maxy)

            # fill background
            background = rendering.make_polygon([(minx, miny),
                                                 (minx, maxy),
                                                 (maxx, maxy),
                                                 (maxx, miny)],
                                                filled=True)
            background.set_color(*BLACK)
            self.viewer.add_geom(background)

            # display airspace
            sector = rendering.make_polygon(self.airspace.polygon.boundary.coords, filled=False)
            sector.set_linewidth(1)
            sector.set_color(*WHITE)
            self.viewer.add_geom(sector)

        # add current positions
        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            if i in self.conflicts:
                color = RED
            elif i in self.warning:
                color = ORANGE
            else:
                color = GREEN

            circle = rendering.make_circle(radius=self.min_distance / 2.0,
                                           res=10,
                                           filled=False)
            circle.add_attr(rendering.Transform(translation=(f.position.x,
                                                             f.position.y)))
            circle.set_color(*color)
            plan = LineString([f.position, f.target])
            self.viewer.draw_polyline(plan.coords, linewidth=1, color=color)
            prediction = LineString([f.position, f.prediction])
            self.viewer.draw_polyline(prediction.coords, linewidth=4, color=color)

            self.viewer.add_onetime(circle)

        self.viewer.render()

    def close(self) -> None:
        """
        Closes the viewer
        :return:
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

