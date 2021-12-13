"""

Created: 2021
Contributors: Nigel Swab
"""
import errno
import os
import re
import sys
from datetime import datetime
from math import isnan
from pprint import pprint
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as df
from pandas import Series
from scipy import stats

import vehicle_calculations as calc
from lib.UnitConversions import UnitConversion
from lib.fit_equations import LinearFits
from lib.print_colour import printy
from vehicle_model import VehicleParameters
from yaw_moment_sweep_limits import VehicleSweepLimits


# style.use('seaborn')
# TODO: Use filters and loc to better sort through data


class YawMomentDiagram:
    __slots__ = [
        "speed_mps",
        "chassis_slip_angles_rad",
        "steering_rack_travels_mm",
        "lat_accel_tolerance_mps2",
        "relaxation_parameter",
        "assume_symmetric_results",
    ]

    speed_mps: float
    chassis_slip_angles_rad: np.ndarray
    steering_rack_travels_mm: np.ndarray
    lat_accel_tolerance_mps2: float
    relaxation_parameter: float
    assume_symmetric_results: bool

    def __init__(
            self,
            car: VehicleParameters,
            speed_kph: float,
            chassis_slip_angle_limit_deg: float,
            chassis_slip_steps: int,
            steering_wheel_angle_limit_deg: float,
            steering_steps: int,
            lat_accel_tolerance_mps2: float,
            relaxation_parameter: float = 0.8,
            assume_symmetric_results=True,
    ):
        # define instance attributes
        self.speed_mps = UnitConversion.kph_to_mps(speed_kph)
        self.lat_accel_tolerance_mps2 = lat_accel_tolerance_mps2
        self.relaxation_parameter = relaxation_parameter
        self.assume_symmetric_results = assume_symmetric_results

        # ensure an odd number of steps so there is a data point at 0 steer and 0 slip
        steering_steps = steering_steps + 1 if not (steering_steps % 2) else steering_steps
        chassis_slip_steps = chassis_slip_steps + 1 if not chassis_slip_steps % 2 else chassis_slip_steps

        # convert input units to relevant SI
        chassis_slip_angle_limits_rad = np.deg2rad(chassis_slip_angle_limit_deg) * -1
        steering_wheel_angle_limits_rad = np.deg2rad(steering_wheel_angle_limit_deg)
        steering_rack_limits_mm = steering_wheel_angle_limits_rad * car.steering_rack_speed_mmprad

        # define sweep stops depending on assumptions
        chassis_slip_angle_stop = 0 if assume_symmetric_results else -chassis_slip_angle_limits_rad
        steering_rack_stop = -steering_rack_limits_mm

        # define instance attributes containing sweeps of chassis slip angles and steering
        self.chassis_slip_angles_rad = np.linspace(
            start=chassis_slip_angle_limits_rad,
            stop=chassis_slip_angle_stop,
            num=(int(np.round(chassis_slip_steps / 2)) if assume_symmetric_results else int(chassis_slip_steps)),
        )

        # prepare non-linear rack travel to avoid stacking at tire saturation
        distribution = stats.norm(loc=0, scale=np.round(steering_rack_limits_mm / 2))
        bounds_for_range = distribution.cdf([-steering_rack_limits_mm, steering_rack_limits_mm])
        pp = np.linspace(*bounds_for_range, num=steering_steps)
        self.steering_rack_travels_mm = distribution.ppf(pp)

        # self.steering_rack_travels_mm = np.linspace(
        #     start=steering_rack_limits_mm, stop=steering_rack_stop, num=int(steering_wheel_angle_steps))

    def yaw_moment_calc_iso(
            self,
            car: VehicleParameters,
            lat_accel_convergence_tolerance_mps2: float,
            relaxation_parameter: float,
            long_accel: float = 0,
            max_convergence_iters: int = 1000,
    ) -> df:

        print("\nCalculating yaw moment diagram values...\n")
        # calculate wheel loads and resulting bump from aero forces
        aero_wheel_load_f_n, aero_wheel_load_r_n = calc.aerodynamic_wheel_loads_n(speed_mps=self.speed_mps, car=car)
        bump_aero_f_m = calc.suspension_bump_m(wheel_load_n=aero_wheel_load_f_n, wheelrate_npm=car.wheelrate_f_npm)
        bump_aero_r_m = calc.suspension_bump_m(wheel_load_n=aero_wheel_load_r_n, wheelrate_npm=car.wheelrate_r_npm)

        # TODO: longitudinal forces to calculate bumps/heave
        # long_accel_wheel_load_f_n, long_accel_wheel_load_r_n = \
        #   calc.longitudinal_load_transfer_calculation()
        # bump_long_accel_f_m = \
        #   calc.suspension_bump_m(wheel_load_n=long_accel_wheel_load_f_n, wheelrate_npm=car.wheelrate_f_npm)
        # bump_long_accel_r_m = \
        #   calc.suspension_bump_m(wheel_load_n=long_accel_wheel_load_r_n, wheelrate_npm=car.wheelrate_r_npm)

        # total bump travel
        bump_f_m = bump_aero_f_m
        bump_r_m = bump_aero_r_m

        # update roll centers and corresponding lateral load transfer sensitivity
        car.roll_center_height_f_m = calc.roll_center_height_m(
            roll_center_gain=car.roll_center_gain_bump_f,
            bump_m=bump_aero_f_m,
            static_roll_center_height_m=car.roll_center_height_f_m,
        )
        car.roll_center_height_r_m = calc.roll_center_height_m(
            roll_center_gain=car.roll_center_gain_bump_r,
            bump_m=bump_aero_r_m,
            static_roll_center_height_m=car.roll_center_height_r_m,
        )
        # TODO: Add change in cg from bummp (maybe)
        car.lateral_load_transfer_sensitivity_definitions()

        # create a list of dicts to be converted into a dataframe once all data is calculated
        yaw_moment_output_data = []  # empty list for result dictionaries
        lat_accel_mps2 = [0]  # initial conditions
        yaw_rate_radps = 0  # initial conditions
        roll_angle_rad = 0  # initial conditions
        convergence_iters = 0  # initial conditions
        tire_lat_forces_f_n = 0
        tire_lat_forces_r_n = 0
        mz_tires_nm = 0
        slip_ratio = 0  # initial condition for no longitudinal force

        for steering_rack_travel_mm in self.steering_rack_travels_mm:
            for chassis_slip_angle_rad in self.chassis_slip_angles_rad:
                rel_lat_accel_error_mps2 = 1
                yaw_rate_radps = 0
                velocity_x_mps = self.speed_mps * np.cos(chassis_slip_angle_rad)
                velocity_y_mps = self.speed_mps * np.sin(chassis_slip_angle_rad)

                for index in range(1, max_convergence_iters + 1):

                    # calculate wheel load transfer from lateral acceleration
                    (
                        lat_accel_load_transfer_fl_n,
                        lat_accel_load_transfer_rl_n,
                    ) = calc.lateral_load_transfer_left_calculation(lat_accel_mps2=lat_accel_mps2[index - 1], car=car)

                    # calculate all wheel loads
                    wheel_load_fl_n = car.static_wheel_load_f_n + aero_wheel_load_f_n + lat_accel_load_transfer_fl_n
                    wheel_load_fr_n = car.static_wheel_load_f_n + aero_wheel_load_f_n - lat_accel_load_transfer_fl_n
                    wheel_load_rl_n = car.static_wheel_load_r_n + aero_wheel_load_r_n + lat_accel_load_transfer_rl_n
                    wheel_load_rr_n = car.static_wheel_load_r_n + aero_wheel_load_r_n - lat_accel_load_transfer_rl_n

                    # calculate roll angle
                    roll_angle_rad = calc.roll_angle_calculation(
                        lateral_accel_mps2=lat_accel_mps2[index - 1], roll_gradient_radpg=car.roll_gradient_radpg
                    )

                    # ensure no negative wheel loads
                    # TODO: likely a more accurate way of distributing load
                    #  - should also consider suspension travel with roll
                    wheel_load_fl_n, wheel_load_fr_n = calc.wheel_load_check(
                        wheel_load_l_n=wheel_load_fl_n, wheel_load_r_n=wheel_load_fr_n
                    )
                    wheel_load_rl_n, wheel_load_rr_n = calc.wheel_load_check(
                        wheel_load_l_n=wheel_load_rl_n, wheel_load_r_n=wheel_load_rr_n
                    )

                    # calculate wheel angles
                    (
                        incln_angle_fl_rad,
                        incln_angle_rl_rad,
                        toe_fl_rad,
                        toe_rl_rad,
                    ) = car.kinematics_l.calculate_wheel_angles(
                        bump_travel_f_m=bump_f_m,
                        bump_travel_r_m=bump_r_m,
                        rack_travel_mm=steering_rack_travel_mm,
                        roll_angle_rad=roll_angle_rad,
                        static_incln_f_rad=car.static_incln_fl_rad,
                        static_incln_r_rad=car.static_incln_rl_rad,
                        static_toe_f_rad=car.static_toe_fl_rad,
                        static_toe_r_rad=car.static_toe_rl_rad,
                    )
                    (
                        incln_angle_fr_rad,
                        incln_angle_rr_rad,
                        toe_fr_rad,
                        toe_rr_rad,
                    ) = car.kinematics_r.calculate_wheel_angles(
                        bump_travel_f_m=bump_f_m,
                        bump_travel_r_m=bump_r_m,
                        rack_travel_mm=steering_rack_travel_mm,
                        roll_angle_rad=roll_angle_rad,
                        static_incln_f_rad=-car.static_incln_fl_rad,
                        static_incln_r_rad=-car.static_incln_rl_rad,
                        static_toe_f_rad=-car.static_toe_fl_rad,
                        static_toe_r_rad=-car.static_toe_rl_rad,
                    )

                    # TODO: determine why offset for k_toe_f is an issue
                    toe_fl_rad = (
                        car.static_toe_fl_rad if chassis_slip_angle_rad and steering_rack_travel_mm == 0 else toe_fl_rad
                    )
                    toe_fr_rad = (
                        -car.static_toe_fl_rad
                        if chassis_slip_angle_rad and steering_rack_travel_mm == 0
                        else toe_fr_rad
                    )
                    toe_rl_rad = (
                        car.static_toe_rl_rad if chassis_slip_angle_rad and steering_rack_travel_mm == 0 else toe_rl_rad
                    )
                    toe_rr_rad = (
                        -car.static_toe_rl_rad
                        if chassis_slip_angle_rad and steering_rack_travel_mm == 0
                        else toe_rr_rad
                    )

                    # calculate slip angles
                    slip_angle_fl_rad = (velocity_y_mps + yaw_rate_radps * car.cg_to_f_wheels_dist_m) / (
                            velocity_x_mps - yaw_rate_radps * car.track_f_m / 2
                    ) - toe_fl_rad
                    slip_angle_fr_rad = (velocity_y_mps + yaw_rate_radps * car.cg_to_f_wheels_dist_m) / (
                            velocity_x_mps + yaw_rate_radps * car.track_f_m / 2
                    ) - toe_fr_rad
                    slip_angle_rl_rad = (velocity_y_mps - yaw_rate_radps * car.cg_to_r_wheels_dist_m) / (
                            velocity_x_mps - yaw_rate_radps * car.track_r_m / 2
                    ) - toe_rl_rad
                    slip_angle_rr_rad = (velocity_y_mps - yaw_rate_radps * car.cg_to_r_wheels_dist_m) / (
                            velocity_x_mps + yaw_rate_radps * car.track_r_m / 2
                    ) - toe_rr_rad

                    # calculate wheel forces
                    fy_fl_n, fx_fl_n, mz_fl_nm, mx_fl_nm = car.tire_f.tire_forces(
                        alpha=slip_angle_fl_rad,
                        kappa=slip_ratio,
                        gamma=incln_angle_fl_rad,
                        Fz=wheel_load_fl_n,
                        tire_pressure=car.tire_pressure_f_pa,
                    )
                    fy_fr_n, fx_fr_n, mz_fr_nm, mx_fr_nm = car.tire_f.tire_forces(
                        alpha=slip_angle_fr_rad,
                        kappa=slip_ratio,
                        gamma=incln_angle_fr_rad,
                        Fz=wheel_load_fr_n,
                        tire_pressure=car.tire_pressure_f_pa,
                    )
                    fy_rl_n, fx_rl_n, mz_rl_nm, mx_rl_nm = car.tire_r.tire_forces(
                        alpha=slip_angle_rl_rad,
                        kappa=slip_ratio,
                        gamma=incln_angle_rl_rad,
                        Fz=wheel_load_rl_n,
                        tire_pressure=car.tire_pressure_r_pa,
                    )
                    fy_rr_n, fx_rr_n, mz_rr_nm, mx_rr_nm = car.tire_r.tire_forces(
                        alpha=slip_angle_rr_rad,
                        kappa=slip_ratio,
                        gamma=incln_angle_rr_rad,
                        Fz=wheel_load_rr_n,
                        tire_pressure=car.tire_pressure_r_pa,
                    )

                    # convert and sum tire forces into chassis frame of reference
                    tire_lat_forces_f_n = np.cos(toe_fl_rad) * fy_fl_n + np.cos(toe_fr_rad) * fy_fr_n
                    tire_lat_forces_r_n = np.cos(toe_rl_rad) * fy_rl_n + np.cos(toe_rr_rad) * fy_rr_n

                    # TODO: add longitudinal forces, include those in long and lat forces in chassis frame

                    mz_tires_nm = mz_fl_nm + mz_fr_nm + mz_rl_nm + mz_rr_nm

                    # calculate lateral acceleration using the relaxation parameter
                    lat_accel_mps2.append(
                        (1 - relaxation_parameter) * lat_accel_mps2[index - 1]
                        + relaxation_parameter * ((tire_lat_forces_f_n + tire_lat_forces_r_n) / car.mass_total_kg)
                    )

                    # calculate relative error in lateral acceleration
                    rel_lat_accel_error_mps2 = np.abs(lat_accel_mps2[index] - lat_accel_mps2[index - 1])

                    # calculate new yaw rate for found lateral acceleration
                    yaw_rate_radps = (lat_accel_mps2[index]) / self.speed_mps

                    # break loop if the relative error in lateral acceleration is small enough
                    if rel_lat_accel_error_mps2 < lat_accel_convergence_tolerance_mps2:
                        convergence_iters = index
                        # print live updates for the user
                        sys.stdout.write(
                            f"\r For Steering rack travel: {steering_rack_travel_mm: .0f} mm"
                            f"\tChassis Slip Angl: {np.rad2deg(chassis_slip_angle_rad): .0f}°"
                            f"\tConvergence Iterations = {convergence_iters}"
                        )
                        break

                    if index == max_convergence_iters + 1:
                        printy(
                            f"Iteration with steering angle {steering_rack_travel_mm / car.steering_rack_speed_mmprad}"
                            f"and chassis slip angle {np.rad2deg(chassis_slip_angle_rad)} failed to converge!"
                        )

                yaw_moment_nm = (
                        -car.cg_to_r_wheels_dist_m * tire_lat_forces_r_n
                        + car.cg_to_f_wheels_dist_m * tire_lat_forces_f_n
                        + mz_tires_nm
                )

                # save data into a dictionary and list of dictionaries
                results = {
                    "Steering Wheel Angle [rad]": steering_rack_travel_mm / car.steering_rack_speed_mmprad,
                    "Chassis Slip Angle [rad]": chassis_slip_angle_rad,
                    "Lateral Acceleration [m/s^2]": lat_accel_mps2[-1],
                    "Yaw Rate [rad/s]": yaw_rate_radps,
                    "Corner Radius [m]": np.abs(self.speed_mps ** 2 / lat_accel_mps2[-1]),
                    "Yaw Moment [Nm]": yaw_moment_nm,
                    "Relative Error [m/s^2]": rel_lat_accel_error_mps2,
                    "Front Lateral Force [N]": tire_lat_forces_f_n,
                    "Rear Lateral Force [N]": tire_lat_forces_r_n,
                    "Roll Angle [rad]": roll_angle_rad,
                    "Iterations until Convergence": convergence_iters,
                    "Yaw Acceleration [rad/s^2]": yaw_moment_nm / car.moi_yaw_kgm2,
                    "Velocity [m/s]": self.speed_mps,
                }
                # append results to list output data list
                yaw_moment_output_data.append(results)

                # append negative results to list if results are assumed to be symmetric
                if self.assume_symmetric_results and chassis_slip_angle_rad != 0.0:
                    opp_results = results.copy()
                    for key, value in results.items():
                        opp_results[key] = (
                            -value
                            if (
                                    key not in ["Relative Error [m/s^2]", "Corner Radius [m]", "Velocity [m/s]"]
                                    and chassis_slip_angle_rad != 0
                            )
                            else value
                        )
                    yaw_moment_output_data.append(opp_results)

                # set new initial acceleration estimate as previous max accel
                lat_accel_mps2[0] = 0
                # reset lateral acceleration
                lat_accel_mps2 = [lat_accel_mps2[0]]

        print("\n\nFinished calculating yaw moment diagram values!\n")

        return df(yaw_moment_output_data)


class YawMomentConvertData:
    @classmethod
    def extract_units(cls, yaw_moment_data: df) -> dict[str:str]:
        list_of_units = [str(re.findall(r"\[.*?\]", column)) for column in yaw_moment_data.columns]
        cls.check_unit_consistency(list_of_units=list_of_units)
        return {
            "Moment": "-" if any("-" in unit for unit in list_of_units) else "Nm",
            "Acceleration": "g" if any("[g" in unit for unit in list_of_units) else "m/s^2",
            "Angle": "°" if any("°" in unit for unit in list_of_units) else "rad",
        }

    @classmethod
    def check_unit_consistency(cls, list_of_units: list[str]) -> None:
        if any("°" in unit for unit in list_of_units) and any("rad" in unit for unit in list_of_units):
            raise ValueError(
                "There are multiple units used for angles in this dataframe! Please use\n"
                "convert_yaw_moment_data() from YawMomentConvertData first\n"
            )
        if any("g" in unit for unit in list_of_units) and any("m/s^2" in unit for unit in list_of_units):
            raise ValueError(
                "There are multiple units used for acceleration or moments in this dataframe!\n"
                "Please use convert_yaw_moment_data() from YawMomentConvertData first\n"
            )
        if any("-" in unit for unit in list_of_units) and any("Nm" in unit for unit in list_of_units):
            raise ValueError(
                "There are multiple units used for acceleration or moments in this dataframe!\n"
                "Please use convert_yaw_moment_data() from YawMomentConvertData first\n"
            )

    @classmethod
    def convert_yaw_moment_data(cls, yaw_moment_data: df, car: VehicleParameters, to_deg: bool, to_normalized: bool):
        # convert and/or normalize data if needed
        yaw_moment_data = (
            cls._convert_yaw_moment_data_to_deg(yaw_moment_data=yaw_moment_data)
            if to_deg
            else cls._convert_yaw_moment_data_to_rad(yaw_moment_data=yaw_moment_data)
        )
        yaw_moment_data = (
            cls._normalize_data(yaw_moment_data=yaw_moment_data, car=car)
            if to_normalized
            else cls._denormalize_data(yaw_moment_data=yaw_moment_data, car=car)
        )

    @classmethod
    def _convert_yaw_moment_data_to_deg(cls, yaw_moment_data: df) -> df:
        for column in yaw_moment_data:
            if "[rad" in column:
                yaw_moment_data[column] = np.rad2deg(yaw_moment_data[column])
                yaw_moment_data.rename(columns={column: column.replace("rad", "°")}, inplace=True)
            if "/rad" in column:
                yaw_moment_data[column] = np.deg2rad(yaw_moment_data[column])
                yaw_moment_data.rename(columns={column: column.replace("rad", "°")}, inplace=True)
            if "_rad" in column:
                yaw_moment_data[column] = np.deg2rad(yaw_moment_data[column])
                yaw_moment_data.rename(columns={column: column.replace("_rad", "°")}, inplace=True)
        return yaw_moment_data

    @classmethod
    def _convert_yaw_moment_data_to_rad(cls, yaw_moment_data: df) -> df:
        for column in yaw_moment_data:
            if "[°" in column:
                yaw_moment_data[column] = np.deg2rad(yaw_moment_data[column])
                yaw_moment_data.rename(columns={column: column.replace("[°", "[rad")}, inplace=True)
            if "/°" in column:
                yaw_moment_data[column] = np.rad2deg(yaw_moment_data[column])
                yaw_moment_data.rename(columns={column: column.replace("/°", "/rad")}, inplace=True)
            if "_°" in column:
                yaw_moment_data[column] = np.rad2deg(yaw_moment_data[column])
                yaw_moment_data.rename(columns={column: column.replace("_°", "_rad")}, inplace=True)
        return yaw_moment_data

    @classmethod
    def _normalize_data(cls, yaw_moment_data: df, car: VehicleParameters) -> df:
        for column in yaw_moment_data:
            if "[m/s^2" in column:  # convert to g-force
                yaw_moment_data[column] = yaw_moment_data[column] / 9.81
                yaw_moment_data.rename(columns={column: column.replace("m/s^2", "g")}, inplace=True)
            if "/m/s^2" in column:
                yaw_moment_data[column] = yaw_moment_data[column] * 9.81
                yaw_moment_data.rename(columns={column: column.replace("m/s^2", "g")}, inplace=True)
            if "Nm" in column:  # normalize yaw moment
                yaw_moment_data[column] = yaw_moment_data[column] / (car.weight_total_n * car.wheelbase_m)
                yaw_moment_data.rename(columns={column: column.replace("Nm", "-")}, inplace=True)
        return yaw_moment_data

    @classmethod
    def _denormalize_data(cls, yaw_moment_data: df, car: VehicleParameters) -> df:
        for column in yaw_moment_data:
            if "[g" in column:  # convert to g-force
                yaw_moment_data[column] = yaw_moment_data[column] * 9.81
                yaw_moment_data.rename(columns={column: column.replace("[g", "[m/s^2")}, inplace=True)
            if "/g]" in column:
                yaw_moment_data[column] = yaw_moment_data[column] / 9.81
                yaw_moment_data.rename(columns={column: column.replace("/g]", "/m/s^2]")}, inplace=True)
            if "-" in column:  # normalize yaw moment
                yaw_moment_data[column] = yaw_moment_data[column] * (car.weight_total_n * car.wheelbase_m)
                yaw_moment_data.rename(columns={column: column.replace("-", "Nm")}, inplace=True)
        return yaw_moment_data

    @classmethod
    def convert_velocity_to_kph(cls, yaw_moment_data: df) -> df:
        for column in yaw_moment_data:
            if "m/s" in column:
                yaw_moment_data[column] = UnitConversion.mps_to_kph(yaw_moment_data[column])
                yaw_moment_data.rename(columns={column: column.replace("m/s", "km/hr")}, inplace=True)
        return yaw_moment_data

    @classmethod
    def convert_velocity_to_mps(cls, yaw_moment_data: df) -> df:
        for column in yaw_moment_data:
            if "km/hr" in column:
                yaw_moment_data[column] = UnitConversion.kph_to_mps(yaw_moment_data[column])
                yaw_moment_data.rename(columns={column: column.replace("km/hr", "m/s")}, inplace=True)
        return yaw_moment_data


class YawMomentAnalysis(YawMomentConvertData):
    yaw_moment_data: df
    units: dict

    def __init__(self, yaw_moment_data: df, to_deg: bool, to_normalized: bool, car: VehicleParameters):
        self.yaw_moment_data = np.round(yaw_moment_data, decimals=8)
        self.is_normalized = to_normalized

        self.units = {
            "Moment": "-" if to_normalized else "Nm",
            "Acceleration": "g" if to_normalized else "m/s^2",
            "Angle": "°" if to_deg else "rad",
        }

        # convert and/or normalize data if needed
        self.convert_yaw_moment_data(
            yaw_moment_data=self.yaw_moment_data, car=car, to_deg=to_deg, to_normalized=to_normalized
        )

        # method title says it all
        self.calculate_performance_indicators()

    def calculate_performance_indicators(self):
        self.calculate_control_accel_indices()
        self.calculate_control_moment_indices()
        self.calculate_static_directional_stability_indices()
        self.calculate_yaw_stability_indices()
        self.fill_nans()

    def fill_nans(self):
        is_nan = self.yaw_moment_data.isnull()
        row_has_nan = is_nan.any(axis=1)
        rows_w_nan = self.yaw_moment_data[row_has_nan]

        for index, row in rows_w_nan.iterrows():
            chassis_slip = self.yaw_moment_data[f"Chassis Slip Angle [{self.units['Angle']}]"].loc[index]
            steer_angle = self.yaw_moment_data[f"Steering Wheel Angle [{self.units['Angle']}]"].loc[index]
            chassis_filt_cond = -chassis_slip if chassis_slip else chassis_slip
            steer_filt_cond = -steer_angle if steer_angle else steer_angle
            data_at_opp = self.yaw_moment_data[
                (self.yaw_moment_data[f"Chassis Slip Angle [{self.units['Angle']}]"] == chassis_filt_cond)
                & (self.yaw_moment_data[f"Steering Wheel Angle [{self.units['Angle']}]"] == steer_filt_cond)
                ]
            for column, value in row.items():
                new_value = data_at_opp[column].iloc[0] if isnan(value) else value
                self.yaw_moment_data.at[index, column] = new_value

    def calculate_static_directional_stability_indices(self) -> None:
        """
        slope of the yaw moment given a change in chassis slip angle for constant steer angle (dN/dBeta)
        Also referred to as directional spring/weathercock effect as described on pg. 228 in
        Race Car Vehicle Dynamics
        """
        self.yaw_moment_data.sort_values(
            [f"Steering Wheel Angle [{self.units['Angle']}]", f"Chassis Slip Angle [{self.units['Angle']}]"],
            inplace=True,
        )
        list_of_steer_angles = np.round(
            self.yaw_moment_data[f"Steering Wheel Angle [{self.units['Angle']}]"], decimals=8
        ).unique()

        static_directional_stability_nmpdeg = []
        for steer_angle in list_of_steer_angles:
            stability_index_data = self.yaw_moment_data[
                np.round(self.yaw_moment_data[f"Steering Wheel Angle [{self.units['Angle']}]"], decimals=8)
                == steer_angle
                ]
            stability_index_data = (
                stability_index_data[
                    [f"Yaw Moment [{self.units['Moment']}]", f"Chassis Slip Angle [{self.units['Angle']}]"]
                ]
                    .copy(deep=True)
                    .diff(axis=0)
            )
            static_directional_stability_nmpdeg.extend(
                stability_index_data[f"Yaw Moment [{self.units['Moment']}]"]
                / stability_index_data[f"Chassis Slip Angle [{self.units['Angle']}]"]
            )

        self.yaw_moment_data[
            f"Static Directional Stability [{self.units['Moment']}/{self.units['Angle']}]"
        ] = static_directional_stability_nmpdeg

    def calculate_control_accel_indices(self) -> None:
        """
        Slope of lateral acceleration at a given chassis slip angle (dAy/ddelta)
        Indicates turn-in capability of the vehicle
        """
        self.yaw_moment_data.sort_values(
            [f"Chassis Slip Angle [{self.units['Angle']}]", f"Steering Wheel Angle [{self.units['Angle']}]"],
            inplace=True,
        )
        list_of_chassis_slip_angles = np.round(
            self.yaw_moment_data[f"Chassis Slip Angle [{self.units['Angle']}]"], decimals=8
        ).unique()

        control_index = []
        for chassis_slip_angle in list_of_chassis_slip_angles:
            control_index_data = self.yaw_moment_data[
                np.round(self.yaw_moment_data[f"Chassis Slip Angle [{self.units['Angle']}]"], decimals=8)
                == chassis_slip_angle
                ]
            control_index_data = (
                control_index_data[
                    [
                        f"Lateral Acceleration [{self.units['Acceleration']}]",
                        f"Steering Wheel Angle [{self.units['Angle']}]",
                    ]
                ]
                    .copy(deep=True)
                    .diff(axis=0)
            )
            control_index.extend(
                control_index_data[f"Lateral Acceleration [{self.units['Acceleration']}]"]
                / control_index_data[f"Steering Wheel Angle [{self.units['Angle']}]"]
            )

        self.yaw_moment_data[
            f"Control Acceleration Index [{self.units['Acceleration']}/{self.units['Angle']}]"
        ] = control_index

    def calculate_control_moment_indices(self):
        """
        Slope of yaw moment at a given chassis slip angle (dN/ddelta)
        Indicates turn-in capability of the vehicle
        """
        self.yaw_moment_data.sort_values(
            [f"Chassis Slip Angle [{self.units['Angle']}]", f"Steering Wheel Angle [{self.units['Angle']}]"],
            inplace=True,
        )
        list_of_chassis_slip_angles = np.round(
            self.yaw_moment_data[f"Chassis Slip Angle [{self.units['Angle']}]"], decimals=8
        ).unique()

        control_index = []
        for chassis_slip_angle in list_of_chassis_slip_angles:
            control_index_data = self.yaw_moment_data[
                np.round(self.yaw_moment_data[f"Chassis Slip Angle [{self.units['Angle']}]"], decimals=8)
                == chassis_slip_angle
                ]
            control_index_data = (
                control_index_data[
                    [f"Yaw Moment [{self.units['Moment']}]", f"Steering Wheel Angle [{self.units['Angle']}]"]
                ]
                    .copy(deep=True)
                    .diff(axis=0)
            )
            control_index.extend(
                control_index_data[f"Yaw Moment [{self.units['Moment']}]"]
                / control_index_data[f"Steering Wheel Angle [{self.units['Angle']}]"]
            )

        self.yaw_moment_data[f"Control Moment Index [{self.units['Moment']}/{self.units['Angle']}]"] = control_index

    def calculate_yaw_stability_indices(self):
        """
        Slope of yaw moment curve (dN/dAy) for constant steer
        """
        self.yaw_moment_data.sort_values(
            [f"Steering Wheel Angle [{self.units['Angle']}]", f"Chassis Slip Angle [{self.units['Angle']}]"],
            inplace=True,
        )
        list_of_steer_angles = np.round(
            self.yaw_moment_data[f"Steering Wheel Angle [{self.units['Angle']}]"], decimals=8
        ).unique()

        stability_index = []
        for steer_angle in list_of_steer_angles:
            stability_index_data = self.yaw_moment_data[
                np.round(self.yaw_moment_data[f"Steering Wheel Angle [{self.units['Angle']}]"], decimals=8)
                == steer_angle
                ]
            stability_index_data = (
                stability_index_data[
                    [f"Yaw Moment [{self.units['Moment']}]", f"Lateral Acceleration [{self.units['Acceleration']}]"]
                ]
                    .copy(deep=True)
                    .diff(axis=0)
            )
            stability_index.extend(
                stability_index_data[f"Yaw Moment [{self.units['Moment']}]"]
                / stability_index_data[f"Lateral Acceleration [{self.units['Acceleration']}]"]
            )

        self.yaw_moment_data[
            f"Yaw Stability Index [{self.units['Moment']}/{self.units['Acceleration']}]"
        ] = stability_index

    def yaw_moment_data_at_key_points(self) -> list[dict]:
        max_lat_accel_index = self.yaw_moment_data[f"Lateral Acceleration [{self.units['Acceleration']}]"].idxmax()
        min_steer_and_slip_index = (
            np.abs(
                self.yaw_moment_data[
                    [f"Steering Wheel Angle [{self.units['Angle']}]", f"Chassis Slip Angle [{self.units['Angle']}]"]
                ]
            ).sum(axis=1)
        ).idxmin()

        yaw_moment_data_at_limit = self.yaw_moment_data.loc[max_lat_accel_index].to_dict()
        yaw_moment_data_at_limit["Key Point"] = "Limit"
        yaw_moment_data_at_straight = self.yaw_moment_data.loc[min_steer_and_slip_index].to_dict()
        yaw_moment_data_at_straight["Key Point"] = "Straight"
        yaw_moment_data_at_trim = self.trimmed_right()
        yaw_moment_data_at_trim["Key Point"] = "Trimmed Limit"

        return [yaw_moment_data_at_limit, yaw_moment_data_at_straight, yaw_moment_data_at_trim]

    def key_performance_indicators(self, key_point_data: list[dict]) -> dict[str:float]:
        yaw_moment_data_at_limit, yaw_moment_data_at_straight, yaw_moment_data_at_trim = key_point_data

        return {
            f"Max Lateral Acceleration [{self.units['Acceleration']}]": yaw_moment_data_at_limit[
                f"Lateral Acceleration [{self.units['Acceleration']}]"
            ],
            f"Spin Tendency [{self.units['Moment']}]": yaw_moment_data_at_limit[f"Yaw Moment [{self.units['Moment']}]"],
            f"Steering Sensitivity [{self.units['Moment']}/{self.units['Angle']}]": yaw_moment_data_at_straight[
                f"Control Moment Index [{self.units['Moment']}/{self.units['Angle']}]"
            ],
            f"Stability Index [{self.units['Moment']}/{self.units['Angle']}]": yaw_moment_data_at_trim[
                f"Static Directional Stability [{self.units['Moment']}/{self.units['Angle']}]"
            ],
            f"Max Roll Angle [{self.units['Angle']}]": yaw_moment_data_at_limit[f"Roll Angle [{self.units['Angle']}]"],
            "Minimum Radius [m]": self.yaw_moment_data["Corner Radius [m]"].min(),
        }

    def trimmed_right(self) -> dict:
        """
        Add documentation here: -> finds two points on a steer or slip line that cross 0 yaw moment line,
        linearly fits a line to each dataframe column, and calculates its value at 0 yaw
                        -> May be better/more efficient to linearly interpolate
        """
        max_accel_index = self.yaw_moment_data[f"Lateral Acceleration [{self.units['Acceleration']}]"].idxmax()
        yaw_at_max = self.yaw_moment_data[f"Yaw Moment [{self.units['Moment']}]"].loc[max_accel_index]

        index_below_0_yaw, index_above_0_yaw = (
            self.steer_angle_lat_accel_intercepts() if yaw_at_max <= 0 else self.chassis_slip_lat_accel_intercepts()
        )

        yaw_moment_data_at_trim = {}
        min_positive_yaw = self.yaw_moment_data[f"Yaw Moment [{self.units['Moment']}]"].loc[index_above_0_yaw]
        min_negative_yaw = self.yaw_moment_data[f"Yaw Moment [{self.units['Moment']}]"].loc[index_below_0_yaw]
        for column in self.yaw_moment_data.columns:
            # TODO: look into interpolating instead
            dependent_data_w_neg_yaw = self.yaw_moment_data[column].loc[index_below_0_yaw]
            dependent_data_w_pos_yaw = self.yaw_moment_data[column].loc[index_above_0_yaw]
            trim_lin_fit = LinearFits(fit_type="linear1", fit_parameter=column)
            trim_lin_fit.fit(
                Series([min_negative_yaw, min_positive_yaw]),
                Series([dependent_data_w_neg_yaw, dependent_data_w_pos_yaw]),
            )
            yaw_moment_data_at_trim[column] = trim_lin_fit.calculate(0)

        return yaw_moment_data_at_trim

    def chassis_slip_lat_accel_intercepts(self) -> list[int, int]:
        """
        If yaw moment is positive at max accel, a line of constant chassis slip will likely be the
        right most point (with the highest lateral acceleration)
        """
        list_of_const_slip_line_data = self.extract_const_slip_lines(
            yaw_moment_data=self.yaw_moment_data, units=self.units
        )

        pos_moment_index = 0
        neg_moment_index = 0
        combined_max_accel = 0
        max_accel_w_neg_moment_index = 0
        max_accel_w_pos_moment_index = 0

        for const_slip_line_data in list_of_const_slip_line_data:
            # TODO: Could probs generalize and extract this loop out to be used for both methods
            const_slip_line_data = const_slip_line_data.sort_values(f"Yaw Moment [{self.units['Moment']}]")

            for index, yaw_moment in const_slip_line_data[f"Yaw Moment [{self.units['Moment']}]"].items():
                neg_moment_index = index if yaw_moment <= 0 else neg_moment_index
                if yaw_moment > 0:
                    pos_moment_index = index
                    break
            const_slip_line_lat_accel = self.yaw_moment_data[f"Lateral Acceleration [{self.units['Acceleration']}]"]
            combined_accel = (
                    const_slip_line_lat_accel.loc[neg_moment_index] + const_slip_line_lat_accel.loc[pos_moment_index]
            )
            if combined_accel > combined_max_accel:
                combined_max_accel = combined_accel
                max_accel_w_neg_moment_index = neg_moment_index
                max_accel_w_pos_moment_index = pos_moment_index

        return [max_accel_w_neg_moment_index, max_accel_w_pos_moment_index]

    def steer_angle_lat_accel_intercepts(self) -> list[int, int]:
        """
        If yaw moment is negative at max accel, a line of constant chassis slip will likely be the
        right most point (with the highest lateral acceleration)
        """
        list_of_const_steer_line_data = self.extract_const_steer_lines(
            yaw_moment_data=self.yaw_moment_data, units=self.units
        )
        lateral_acceleration = self.yaw_moment_data[f"Lateral Acceleration [{self.units['Acceleration']}]"]

        pos_moment_index = 0
        neg_moment_index = 0
        combined_max_accel = 0
        max_accel_w_neg_moment_index = 0
        max_accel_w_pos_moment_index = 0

        for const_steer_line_data in list_of_const_steer_line_data:
            # TODO: Could probs generalize and extract this loop out to be used for both methods
            const_steer_line_data = const_steer_line_data.sort_values(f"Yaw Moment [{self.units['Moment']}]")

            for index, yaw_moment in const_steer_line_data[f"Yaw Moment [{self.units['Moment']}]"].items():
                neg_moment_index = index if yaw_moment <= 0 else neg_moment_index
                if yaw_moment > 0:
                    pos_moment_index = index
                    break

            combined_accel = lateral_acceleration.loc[neg_moment_index] + lateral_acceleration.loc[pos_moment_index]
            if combined_accel > combined_max_accel:
                combined_max_accel = combined_accel
                max_accel_w_neg_moment_index = neg_moment_index
                max_accel_w_pos_moment_index = pos_moment_index

        return [max_accel_w_neg_moment_index, max_accel_w_pos_moment_index]

    @classmethod
    def extract_const_slip_lines(cls, yaw_moment_data: df, units: Optional[dict[str, str]]) -> list[df]:
        if not units:
            cls.extract_units(yaw_moment_data=yaw_moment_data)

        yaw_moment_data.sort_values(
            [f"Chassis Slip Angle [{units['Angle']}]", f"Steering Wheel Angle [{units['Angle']}]"], inplace=True
        )
        chassis_slip_angles = np.round(yaw_moment_data[f"Chassis Slip Angle [{units['Angle']}]"], decimals=8).unique()

        const_slip_lines_list = [df()]
        for chassis_slip in chassis_slip_angles:
            const_slip_line_data = yaw_moment_data[
                np.round(yaw_moment_data[f"Chassis Slip Angle [{units['Angle']}]"], decimals=8) == chassis_slip
                ]
            const_slip_line_data = const_slip_line_data.sort_values(f"Steering Wheel Angle [{units['Angle']}]")
            const_slip_lines_list.append(const_slip_line_data)
        del const_slip_lines_list[0]
        return const_slip_lines_list

    @classmethod
    def extract_const_steer_lines(cls, yaw_moment_data: df, units: Optional[dict[str, str]]) -> list[df]:
        if not units:
            cls.extract_units(yaw_moment_data=yaw_moment_data)

        yaw_moment_data.sort_values(
            [f"Steering Wheel Angle [{units['Angle']}]", f"Chassis Slip Angle [{units['Angle']}]"], inplace=True
        )
        steering_wheel_angles_rad = np.round(
            yaw_moment_data[f"Steering Wheel Angle [{units['Angle']}]"], decimals=8
        ).unique()

        const_steer_lines_list = [df()]
        for steer_angle in steering_wheel_angles_rad:
            const_steer_line_data = yaw_moment_data[
                np.round(yaw_moment_data[f"Steering Wheel Angle [{units['Angle']}]"], decimals=8) == steer_angle
                ]
            const_steer_line_data = const_steer_line_data.sort_values(f"Chassis Slip Angle [{units['Angle']}]")
            const_steer_lines_list.append(const_steer_line_data)
        del const_steer_lines_list[0]
        return const_steer_lines_list


class YawMomentPlotting(YawMomentAnalysis):
    @classmethod
    def simple_yaw_moment_diagram(
            cls,
            yaw_moment_data: df,
            car: VehicleParameters,
            to_normalized: bool = False,
            to_deg: bool = False,
            to_kph: bool = False,
    ):

        units = {
            "Moment": "-" if to_normalized else "Nm",
            "Acceleration": "g" if to_normalized else "m/s^2",
            "Angle": "°" if to_deg else "rad",
        }
        angle_decimals_format_str = ".0f" if to_deg else ".2f"

        data_units = cls.extract_units(yaw_moment_data=yaw_moment_data)
        if data_units != units:
            cls.convert_yaw_moment_data(yaw_moment_data, car, to_deg, to_normalized)

        figure, axes = plt.subplots(nrows=1, ncols=1)

        list_of_const_slip_line_data = cls.extract_const_slip_lines(yaw_moment_data=yaw_moment_data, units=units)
        list_of_const_steer_line_data = cls.extract_const_steer_lines(yaw_moment_data=yaw_moment_data, units=units)

        # TODO: Break out function for plotting these
        chassis_slip_max = yaw_moment_data[f"Chassis Slip Angle [{units['Angle']}]"].max()
        for const_slip_line_data in list_of_const_slip_line_data:
            # const_slip_line_data = const_slip_line_data.sort_values(f"Steering Wheel Angle [{units['Angle']}]")
            (line,) = axes.plot(
                const_slip_line_data[f"Lateral Acceleration [{units['Acceleration']}]"],
                const_slip_line_data[f"Yaw Moment [{units['Moment']}]"],
                color="b",
                linewidth=0.5,
            )
            chassis_slip = const_slip_line_data[f"Chassis Slip Angle [{units['Angle']}]"].max()
            if chassis_slip == chassis_slip_max:
                line.set_label(
                    f"Constant Chassis Slip Angle (±{chassis_slip: {angle_decimals_format_str}}{units['Angle']})"
                )

        steer_max = yaw_moment_data[f"Steering Wheel Angle [{units['Angle']}]"].max()
        for const_steer_line_data in list_of_const_steer_line_data:
            # const_steer_line_data = const_steer_line_data.sort_values(f"Chassis Slip Angle [{units['Angle']}]")
            (line,) = axes.plot(
                const_steer_line_data[f"Lateral Acceleration [{units['Acceleration']}]"],
                const_steer_line_data[f"Yaw Moment [{units['Moment']}]"],
                color="r",
                linewidth=0.5,
            )
            steer_angle = const_steer_line_data[f"Steering Wheel Angle [{units['Angle']}]"].max()
            if steer_angle == steer_max:
                line.set_label(
                    f"Constant Steering Wheel Angle (±{steer_angle: {angle_decimals_format_str}}{units['Angle']})"
                )

        speed_mps = yaw_moment_data["Velocity [m/s]"].iloc[-1]
        speed_units = "[km/hr]" if to_kph else "[m/s]"
        title_speed = UnitConversion.mps_to_kph(speed_mps) if to_kph else speed_mps
        plt.title(f"Yaw Moment Diagram at {title_speed: .0f} {speed_units}")
        plt.xlabel(f"Lateral Acceleration [{units['Acceleration']}]")
        plt.ylabel(f"Yaw Moment [{units['Moment']}]")
        plt.minorticks_on()
        plt.grid(which="minor", linewidth=0.2)
        plt.grid(which="major", linewidth=0.4)
        plt.legend(loc="best")
        plt.autoscale()
        mng = plt.get_current_fig_manager()
        mng.set_window_title("Yaw Moment Diagram")
        plt.show(block=False)

    @classmethod
    def plot_sweep_at_key_point(
            cls,
            key_metric_data: df,
            key_point: str,  # 'Limit', 'Trimmed Limit', 'Straight'
            x_axis_metric: str,  # column in key_metric_data
            to_sweep_y_metrics: bool = False,  # column in key_metric_data
            to_save_plot: bool = False,
            to_plot: bool = True,
            single_y_metric: Optional[str] = None,
    ):
        units = YawMomentConvertData.extract_units(yaw_moment_data=key_metric_data)
        x_axis_metric_formatted = cls.format_heading_string(x_axis_metric)

        if units["Angle"] == "°" and "_rad" in x_axis_metric:
            key_metric_data[x_axis_metric] = np.rad2deg(key_metric_data[x_axis_metric])
            x_axis_metric_formatted = x_axis_metric.replace("Rad", " [°]")
            key_metric_data.rename(columns={x_axis_metric: x_axis_metric_formatted}, inplace=True)
        elif units["Angle"] == "rad" and "_rad" in x_axis_metric:
            x_axis_metric_formatted = x_axis_metric.replace("Rad", " [rad]")
            key_metric_data.rename(columns={x_axis_metric: x_axis_metric_formatted}, inplace=True)

        # sort and filter data
        # TODO: Could probably extract this for use elsewhere too
        # TODO:
        key_metric_data.sort_values([x_axis_metric])
        key_metric_data_filter = key_metric_data["Key Point"] == key_point
        key_metric_data = key_metric_data[key_metric_data_filter]
        speed_kph = UnitConversion.mps_to_kph(key_metric_data["Velocity [m/s]"].iloc[0])

        y_axis_metrics = [single_y_metric] if not to_sweep_y_metrics else [
            f"Lateral Acceleration [{units['Acceleration']}]",
            f"Control Moment Index [{units['Moment']}/{units['Angle']}]",
            f"Static Directional Stability [{units['Moment']}/{units['Angle']}]",
            f"Static Directional Stability [{units['Moment']}/{units['Angle']}]",
        ]

        if key_point == "Limit":
            y_axis_metrics.append(f"Yaw Moment [{units['Moment']}]")

        for y_axis_metric in y_axis_metrics:
            y_axis_metric_formatted = cls.format_heading_string(y_axis_metric)
            figure, axes = plt.subplots(nrows=1, ncols=1)
            axes.plot(key_metric_data[x_axis_metric], key_metric_data[y_axis_metric], linewidth=0.5)

            plt.ylabel(y_axis_metric)

            if key_point == "Limit" and y_axis_metric == f"Yaw Moment [{units['Moment']}]":
                y_axis_metric = f"Spin Tendency [{units['Moment']}]"

            plt.suptitle(f"{y_axis_metric} {x_axis_metric_formatted} Sensitivity", y=1.05, fontsize=18)
            plt.title(f"{key_point} @ {speed_kph: .0f} km/hr", fontsize=10)
            plt.xlabel(x_axis_metric_formatted)
            plt.minorticks_on()
            plt.grid(which="minor", linewidth=0.2)
            plt.grid(which="major", linewidth=0.4)
            plt.autoscale()
            mng = plt.get_current_fig_manager()
            mng.set_window_title(f"{x_axis_metric_formatted} Sweep Diagram")

            if to_plot:
                plt.show(block=False)

            if not to_save_plot:
                return

            x_metric_filename = cls.format_filename(x_axis_metric_formatted)
            y_metric_filename = cls.format_filename(y_axis_metric_formatted)
            file_location = cls.get_working_directory()

            save_file_folder = \
                f"/{datetime.today().strftime('%Y-%m-%d')} {key_point}/{y_metric_filename}"
            full_filepath = f"{file_location}/{save_file_folder}/{x_metric_filename} Sweep"
            cls.create_file_dir(filepath=full_filepath)
            plt.savefig(full_filepath)

    @classmethod
    def get_working_directory(cls) -> str:
        return f"{os.getcwd()}"

    @classmethod
    def format_filename(cls, filename) -> str:
        filename.replace('/', 'per')
        filename.replace('°', 'deg')
        return re.sub('\W+', '', filename)

    @classmethod
    def create_file_dir(cls, filepath) -> None:
        if not os.path.exists(os.path.dirname(filepath)):
            try:
                os.makedirs(os.path.dirname(filepath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    @classmethod
    def format_heading_string(cls, string_for_formatting: str) -> str:

        string_formatted = string_for_formatting.replace("_", " ")
        string_formatted = string_formatted.title()
        if "Kg" in string_formatted:
            string_formatted = string_formatted.replace('Kg', "[kg]")
        if " M" in string_formatted:
            string_formatted = string_formatted.replace('M', "[m]")
        if "Pa" in string_formatted:
            string_formatted = string_formatted.replace('Pa', "[Pa]")
        if "Npm" in string_formatted:
            string_formatted = string_formatted.replace('Nmp', "[N/m]")
        if "Nmpdeg" in string_formatted:
            string_formatted = string_formatted.replace('N.pdeg', "[Nm/°]")
        if "Incln" in string_formatted:
            string_formatted = string_formatted.replace('Incln', "Inclination Angle")
        if "Coeff" in string_formatted:
            string_formatted = string_formatted.replace('Coeff', "Coefficient")
        if "Arb" in string_formatted:
            string_formatted = string_formatted.replace('Arb', "Anti-Roll Bar")
        if " F" in string_formatted:
            string_formatted = string_formatted.replace('F', "(Front)")
        if " R" in string_formatted:
            string_formatted = string_formatted.replace('R', "(Rear)")
        if " Fl" in string_formatted:
            string_formatted = string_formatted.replace('Fl', "(Front Left)")
        if " Rl" in string_formatted:
            string_formatted = string_formatted.replace('Rl', "(Rear Left)")

        return string_formatted


def sweep_speed_yaw_moment(
        car: VehicleParameters,
        speed_range_kph: tuple[int, int],
        number_of_steps: int
        ):
    list_of_speeds_kph = np.linspace(start=speed_range_kph[0], stop=speed_range_kph[1],
                                     num=number_of_steps)

    key_point_data = []
    for speed_kph in list_of_speeds_kph:
        ymd = YawMomentDiagram(
            car=car,
            speed_kph=speed_kph,
            chassis_slip_angle_limit_deg=8,
            chassis_slip_steps=16,
            steering_wheel_angle_limit_deg=130,
            lat_accel_tolerance_mps2=0.0001,
            relaxation_parameter=0.8,
            steering_steps=16,
            assume_symmetric_results=True,
        )
        print(f'At {speed_kph} km/hr')
        ymd_data = ymd.yaw_moment_calc_iso(
            car=car, lat_accel_convergence_tolerance_mps2=0.00001, relaxation_parameter=0.8
        )
        yaw_moment_analysis = YawMomentAnalysis(yaw_moment_data=ymd_data, car=car,
                                                to_deg=False, to_normalized=False)
        key_point_data_dicts = yaw_moment_analysis.yaw_moment_data_at_key_points()
        key_point_data.extend(key_point_data_dicts)

    return df(key_point_data)


def sweep_car_param_yaw_moment(
        car: VehicleParameters,
        car_sweep_parameter: str,
        car_sweep_parameter_limits: list,
        car_sweep_steps: int,
        speed_kph: int,
):
    list_of_param_values = np.linspace(
        start=car_sweep_parameter_limits[0], stop=car_sweep_parameter_limits[1], num=car_sweep_steps
    )
    test_car = car
    key_point_data = []
    for param_value in list_of_param_values:
        setattr(test_car, car_sweep_parameter, param_value)
        test_car.define_calculated_vehicle_parameters()
        ymd = YawMomentDiagram(
            car=test_car,
            speed_kph=speed_kph,
            chassis_slip_angle_limit_deg=8,
            chassis_slip_steps=16,
            steering_wheel_angle_limit_deg=130,
            lat_accel_tolerance_mps2=0.0001,
            relaxation_parameter=0.8,
            steering_steps=16,
            assume_symmetric_results=True,
        )
        print(f'For {car_sweep_parameter} at {param_value}')
        ymd_data = ymd.yaw_moment_calc_iso(
            car=car, lat_accel_convergence_tolerance_mps2=0.00001, relaxation_parameter=0.8
        )
        yaw_moment_analysis = YawMomentAnalysis(yaw_moment_data=ymd_data, car=car,
                                                to_deg=False, to_normalized=False)
        key_point_data_dicts = yaw_moment_analysis.yaw_moment_data_at_key_points()

        for dictionary in key_point_data_dicts:
            dictionary[car_sweep_parameter] = param_value
        key_point_data.extend(key_point_data_dicts)

        # For debugging
        # YawMomentPlotting.simple_yaw_moment_diagram(
        #     yaw_moment_data=ymd_data, car=car, to_deg=True, to_normalized=True, to_kph=True
        # )
    return df(key_point_data)


def main(save_data: bool = False):
    car = VehicleParameters()
    ymd = YawMomentDiagram(
        car=car,
        speed_kph=60,
        chassis_slip_angle_limit_deg=8,
        chassis_slip_steps=32,
        steering_wheel_angle_limit_deg=130,
        lat_accel_tolerance_mps2=0.0001,
        relaxation_parameter=0.8,
        steering_steps=32,
        assume_symmetric_results=True,
    )
    ymd_data = ymd.yaw_moment_calc_iso(car=car, lat_accel_convergence_tolerance_mps2=0.00001, relaxation_parameter=0.8)
    yaw_moment_analysis = YawMomentAnalysis(yaw_moment_data=ymd_data, car=car, to_deg=True, to_normalized=True)
    data_at_key_points = yaw_moment_analysis.yaw_moment_data_at_key_points()
    performance_indicators = yaw_moment_analysis.key_performance_indicators(key_point_data=data_at_key_points)
    pprint(performance_indicators, sort_dicts=False)
    YawMomentPlotting.simple_yaw_moment_diagram(
        yaw_moment_data=ymd_data, car=car, to_deg=True, to_normalized=True, to_kph=True
    )
    if save_data:
        filename = r"C:\Users\Nigel Swab\Desktop\sample_data.csv"
        yaw_moment_analysis.yaw_moment_data.to_csv(filename)


if __name__ == "__main__":
    # Uncomment for single yaw moment calc
    # main()

    # Uncomment for yaw moment parameter sweep
    car = VehicleParameters()
    limits = VehicleSweepLimits()

    parameters = []
    for param in limits.__dir__():
        if "__" not in param:
            parameters.append(param)
    # parameters = ["static_toe_fl_rad"]
    for param in parameters:
        key_point_param_sweep_data = sweep_car_param_yaw_moment(
            car=car,
            car_sweep_parameter=param,
            car_sweep_parameter_limits=limits.__getattribute__(param),
            car_sweep_steps=16,
            speed_kph=60,
        )
        for point in ["Limit", "Trimmed Limit", "Straight"]:
            YawMomentPlotting.plot_sweep_at_key_point(key_metric_data=key_point_param_sweep_data,
                                                      key_point=point,
                                                      x_axis_metric=param,
                                                      to_sweep_y_metrics=True,
                                                      to_save_plot=True,
                                                      to_plot=False,
                                                      single_y_metric=None,
                                                      )

    speed_range_kph = (40, 120)
    key_point_speed_sweep_data = sweep_speed_yaw_moment(car=car,
                                                        speed_range_kph=speed_range_kph,
                                                        number_of_steps=32)
    for point in ["Limit", "Trimmed Limit", "Straight"]:
        YawMomentPlotting.plot_sweep_at_key_point(key_metric_data=key_point_speed_sweep_data,
                                                  key_point=point,
                                                  x_axis_metric="Velocity [m/s]",
                                                  to_sweep_y_metrics=True,
                                                  to_save_plot=True,
                                                  to_plot=False,
                                                  single_y_metric=None,
                                                  )

    printy('\n\n\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬', colour='turquoise')
    printy('   ☻ SWEEPS DONE ☻', colour='turquoise')
    printy('▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n\n\n', colour='turquoise')
