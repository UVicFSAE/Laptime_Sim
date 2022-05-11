"""

Created: 2021
Contributors: Nigel Swab
"""

import copy
import errno
import os
import re
import sys
import warnings
from datetime import datetime
from math import isnan
from pprint import pprint
from time import perf_counter
from typing import Optional, Union
from matplotlib import use as plt_use
from dataclasses import dataclass

plt_use('Qt5Agg')  # Need to set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as df
from pandas import Series, concat
from scipy import stats
from matplotlib.pyplot import cm
from matplotlib import colors
# import plotly.express as px


import vehicle_calculations as calc
from lib import UnitConversion
from lib.fit_equations import LinearFits
from lib.print_colour import printy
from vehicle_model import VehicleParameters, VehicleSweepLimits

# TODO: Use filters and loc to better sort through data
# TODO: reduce any repeated variable instance (especially cars) to reduce memory useage
# TODO: Replace lists with tuples, especially for repetitions
# TODO: Remove as many '.' as possible


def create_normal_distr_range(upper_bound: float,
                              lower_bound: float,
                              num_steps: int,
                              mean: float = 0,
                              std_dev: float = 0,) -> tuple[float]:
    """
    Create a tuple of normally distributed values around the mean to be incremented through
    """
    scale = std_dev or upper_bound / 2
    distribution = stats.norm(loc=mean, scale=scale)
    bounds_for_range = distribution.cdf([lower_bound, upper_bound])
    pp = np.linspace(*bounds_for_range, num=num_steps)
    return tuple(distribution.ppf(pp))


def create_list(upper_bound: float,
                lower_bound: float,
                num_steps: int,):
    pass


def create_steering_tuple(steering_wheel_angle_limit_deg: float,
                          steering_steps: int,
                          steering_rack_speed_mmprad: float,
                          to_normal_distr: bool = False
                          ) -> tuple[float]:
    # ensure an odd number of steps so there is a data point at 0 steer
    steering_steps = steering_steps + 1 if not (steering_steps % 2) else steering_steps
    # convert input units to relevant SI
    steering_wheel_angle_limits_rad = np.deg2rad(steering_wheel_angle_limit_deg)
    steering_rack_limits_mm = steering_wheel_angle_limits_rad * steering_rack_speed_mmprad

    if to_normal_distr:
        return create_normal_distr_range(upper_bound=steering_rack_limits_mm,
                                         lower_bound=-steering_rack_limits_mm,
                                         num_steps=steering_steps)
    else:
        return tuple(
            np.linspace(start=-steering_rack_limits_mm, stop=steering_rack_limits_mm, num=steering_steps)
        )

#
# def convert_steering_to_rack_travel
#
# def name_me_pls(chassis_slip_angle_limit_deg: float,
#                 chassis_slip_steps: int,
#
#                 to_normal_distr_chassis: bool = False,
#                 to_normal_distr_steer: bool = False,
#                 ) -> (tuple[float], tuple[float]):
#     # ensure an odd number of steps so there is a data point at 0 steer and 0 slip
#     steering_steps = steering_steps + 1 if not (steering_steps % 2) else steering_steps
#     chassis_slip_steps = chassis_slip_steps + 1 if not chassis_slip_steps % 2 else chassis_slip_steps
#
#     # convert input units to relevant SI
#     chassis_slip_angle_limits_rad = np.deg2rad(chassis_slip_angle_limit_deg) * -1
#     steering_wheel_angle_limits_rad = np.deg2rad(steering_wheel_angle_limit_deg)
#     steering_rack_limits_mm = steering_wheel_angle_limits_rad * car.steering_rack_speed_mmprad
#     pass


@dataclass
class YawMomentSetup:
    # car: VehicleParameters
    # vehicle_speed_mps: float
    # longitudinal_accel_mps: float
    # chassis_slip_angles_rad: tuple[float]
    # steering_rack_travels_mm: tuple[float]
    # lat_accel_tolerance_mps2 = 0.0000001
    # relaxation_parameter = 0.65
    # max_convergence_iterations = 1000
    # assume_symmetric_results = False

    """
    For any given yaw moment diagram, the following must be input by the user to proceed
    """

    car: VehicleParameters
    vehicle_speed_kph: Union[int, float]
    long_accel_bounds_g: Union[tuple[float], float]
    long_accel_steps: int
    chassis_slip_angle_bounds_deg: Union[list[float], float]
    chassis_slip_angle_steps: int
    steering_wheel_angle_bounds_deg: Union[list[float], float]
    steering_wheel_angle_steps: int
    lat_accel_tolerance_mps2: float
    relaxation_parameter: float
    max_convergence_iterations: int
    assume_symmetric_results: bool

    """
    For any given yaw moment diagram, the following must be defined/calculated for yaw moment calculation
    """

    car: VehicleParameters
    vehicle_speed_mps: float
    longitudinal_acceleration_mps: float
    chassis_slip_angles_rad: tuple[float]
    steering_wheel_travels_mm: tuple[float]
    steering_wheel_angle_steps: int
    lat_accel_tolerance_mps2: float
    relaxation_parameter: float
    max_convergence_iterations: int


    def __init__(self,
                 car: VehicleParameters,
                 vehicle_speed_kph: Union[int, float],
                 chassis_slip_angle_limit_deg: Union[int, float],
                 chassis_slip_angle_steps: int,
                 steering_wheel_angle_limit_deg: Union[int, float],
                 steering_wheel_angle_steps: int,
                 lat_accel_tolerance_mps2: float = 0.0000001,
                 relaxation_parameter: float = 0.65,
                 max_convergence_iterations: int = 1000,
                 to_normal_distr_chassis: bool = False,
                 to_normal_distr_steering: bool = False,
                 assume_symmetric_results: bool = False,
                 ):
        self.car = car
        self.vehicle_speed_mps = UnitConversion.kph_to_mps(vehicle_speed_kph)


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    plt.show()
