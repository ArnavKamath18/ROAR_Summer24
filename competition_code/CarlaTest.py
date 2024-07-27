import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper, ManualControlViewer
from typing import List, Type, Optional, Dict, Any
import carla
import numpy as np
import asyncio

class RoarCompetitionRule:
    def __init__(self, waypoints, vehicle, world):
        self.waypoints = waypoints
        self.vehicle = vehicle
        self.world = world
        self._last_vehicle_location = vehicle.get_3d_location()
        self._respawn_location = None
        self._respawn_rpy = None
        self.furthest_waypoints_index = 0

    def initialize_race(self):
        self._last_vehicle_location = self.vehicle.get_3d_location()
        vehicle_location = self._last_vehicle_location
        closest_waypoint_dist = np.inf
        closest_waypoint_idx = 0
        for i, waypoint in enumerate(self.waypoints):
            waypoint_dist = np.linalg.norm(vehicle_location - waypoint.location)
            if waypoint_dist < closest_waypoint_dist:
                closest_waypoint_dist = waypoint_dist
                closest_waypoint_idx = i
        self.waypoints = self.waypoints[closest_waypoint_idx+1:] + self.waypoints[:closest_waypoint_idx+1]
        self.furthest_waypoints_index = 0
        print(f"Total length: {len(self.waypoints)}")
        self._respawn_location = self._last_vehicle_location.copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()

    def lap_finished(self, check_step=5):
        return self.furthest_waypoints_index + check_step >= len(self.waypoints)

    async def tick(self, check_step=15):
        current_location = self.vehicle.get_3d_location()
        delta_vector = current_location - self._last_vehicle_location
        delta_vector_norm = np.linalg.norm(delta_vector)
        delta_vector_unit = (delta_vector / delta_vector_norm) if delta_vector_norm >= 1e-5 else np.zeros(3)

        previous_furthest_index = self.furthest_waypoints_index
        min_dis = np.inf
        min_index = 0
        ending_index = previous_furthest_index + check_step if (previous_furthest_index + check_step <= len(self.waypoints)) else len(self.waypoints)
        for i, waypoint in enumerate(self.waypoints[previous_furthest_index:ending_index]):
            waypoint_delta = waypoint.location - current_location
            projection = np.dot(waypoint_delta, delta_vector_unit)
            projection = np.clip(projection, 0, delta_vector_norm)
            closest_point_on_segment = current_location + projection * delta_vector_unit
            distance = np.linalg.norm(waypoint.location - closest_point_on_segment)
            if distance < min_dis:
                min_dis = distance
                min_index = i
        
        self.furthest_waypoints_index += min_index
        self._last_vehicle_location = current_location
        print(f"Reached waypoint {self.furthest_waypoints_index} at {self.waypoints[self.furthest_waypoints_index].location}")

    async def respawn(self):
        self.vehicle.set_transform(self._respawn_location, self._respawn_rpy)
        self.vehicle.set_linear_3d_velocity(np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        for _ in range(20):
            await self.world.step()
        self._last_vehicle_location = self.vehicle.get_3d_location()
        self.furthest_waypoints_index = 0

class RoarCompetitionSolution:
    def __init__(self, waypoints, agent, camera, location_sensor, velocity_sensor, rpy_sensor, occupancy_map_sensor, collision_sensor):
        self.waypoints = waypoints
        self.agent = agent
        self.camera = camera
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

    async def initialize(self):
        pass

    def get_desired_speed(self, location):
        # Placeholder for desired speed logic
        return 30.0  # desired speed in m/s

    def calculate_curvature(self, waypoints, current_index):
        if current_index + 2 >= len(waypoints):
            return 0.0
        next_wp = waypoints[current_index + 1]
        next_next_wp = waypoints[current_index + 2]
        direction = next_next_wp.location - next_wp.location
        distance = np.linalg.norm(direction)
        if distance == 0:
            return 0.0
        curvature = 2 * np.cross(next_wp.location - waypoints[current_index].location, direction) / (distance ** 2)
        return np.linalg.norm(curvature)