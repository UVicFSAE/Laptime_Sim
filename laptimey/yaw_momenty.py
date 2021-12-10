"""

Created: 2021
Contributors: Nigel Swab
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as df
from scipy import stats

import vehicle_calculations as calc
from lib.UnitConversions import UnitConversion
from lib.print_colour import printy
from vehicle_model import VehicleParameters


# style.use('seaborn')


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
            num=(
                int(np.round(chassis_slip_steps / 2))
                if assume_symmetric_results
                else int(chassis_slip_steps)
            ),
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
                    "Steering Wheel Angle [deg]": np.rad2deg(steering_rack_travel_mm / car.steering_rack_speed_mmprad),
                    "Chassis Slip Angle [rad]": chassis_slip_angle_rad,
                    "Chassis Slip Angle [deg]": np.rad2deg(chassis_slip_angle_rad),
                    "Lateral Acceleration [m/s^2]": lat_accel_mps2[-1],
                    "Normalized Lateral Acceleration [g]": lat_accel_mps2[-1] / 9.81,
                    "Yaw Rate [rad/s]": yaw_rate_radps,
                    "Corner Radius [m]": np.abs(self.speed_mps ** 2 / lat_accel_mps2[-1]),
                    "Yaw Moment [Nm]": yaw_moment_nm,
                    "Normalized Yaw Moment [-]": yaw_moment_nm / (car.weight_total_n * car.wheelbase_m),
                    "Relative Error [m/s^2]": rel_lat_accel_error_mps2,
                    "Front Lateral Force [N]": tire_lat_forces_f_n,
                    "Rear Lateral Force [N]": tire_lat_forces_r_n,
                    "Roll Angle [rad]": roll_angle_rad,
                    "Iterations until Convergence": convergence_iters,
                    "Yaw Acceleration [rad/s^2]": yaw_moment_nm / car.moi_yaw_kgm2,
                }
                # append results to list output data list
                yaw_moment_output_data.append(results)

                # append negative results to list if results are assumed to be symmetric
                if self.assume_symmetric_results:
                    opp_results = results.copy()
                    for key, value in results.items():
                        opp_results[key] = (
                            -value
                            if (
                                key not in ["Relative Error [m/s^2]", "Corner Radius [m]"]
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

    def key_performance_indicators(self, yaw_moment_data: df) -> dict[str:float]:

        self.static_directional_stability_index(yaw_moment_data=yaw_moment_data)
        self.control_force_index(yaw_moment_data=yaw_moment_data)
        self.control_moment_index(yaw_moment_data=yaw_moment_data)
        self.yaw_stability_index(yaw_moment_data=yaw_moment_data)

        # TODO: add interpolation for points

        max_lat_accel_mps2 = yaw_moment_data["Lateral Acceleration [m/s^2]"].max()
        max_accel_index = yaw_moment_data["Lateral Acceleration [m/s^2]"].idxmax()
        return {
            "Max Lateral Acceleration [g]": max_lat_accel_mps2 / 9.81,
            "Max Lateral Acceleration [m/s^2]": max_lat_accel_mps2,
            "Minimum Radius [m]": yaw_moment_data["Corner Radius [m]"].min(),
            "Spin Tendency [Nm]": yaw_moment_data["Yaw Moment [Nm]"].iloc[max_accel_index],
            "Spin Tendency [-]": yaw_moment_data["Normalized Yaw Moment [-]"].iloc[max_accel_index],
            "Max Yaw Moment [Nm]": yaw_moment_data["Yaw Moment [Nm]"].max(),
            "Max Roll Angle [deg]": np.rad2deg(yaw_moment_data["Roll Angle [rad]"]),
        }

    @staticmethod
    def find_key_df_indices(yaw_moment_data: df) -> tuple[int, int, int]:
        max_lat_accel_index = yaw_moment_data["Lateral Acceleration [m/s^2"].idxmax()
        max_trim_lat_index = 0  # TODO: Figure out a good way to do this
        min_steer_and_slip_index = np.abs(
            yaw_moment_data[["Steering Wheel Angle [rad]", "Chassis Slip Angle [rad]"]].sum(axis=1)
        ).idxmin()
        return max_lat_accel_index, max_trim_lat_index, min_steer_and_slip_index

    def calculate_performance_indexes(self, yaw_moment_data: df):
        self.control_force_index(yaw_moment_data=yaw_moment_data)
        self.control_moment_index(yaw_moment_data=yaw_moment_data)
        self.static_directional_stability_index(yaw_moment_data=yaw_moment_data)
        self.yaw_stability_index(yaw_moment_data=yaw_moment_data)
        yaw_moment_data.sort_index(inplace=True)
        print("hi")

    @staticmethod
    def static_directional_stability_index(yaw_moment_data: df):
        """
        slope of the yaw moment given a change in chassis slip angle for constant steer angle (dN/dBeta)
        Also referred to as directional spring/weathercock effect as described on pg. 228 in
        Race Car Vehicle Dynamics
        """
        yaw_moment_data.sort_values(["Steering Wheel Angle [rad]", "Chassis Slip Angle [rad]"], inplace=True)
        list_of_steer_angles = np.round(yaw_moment_data["Steering Wheel Angle [rad]"], decimals=8).unique()

        static_directional_stability_nmpdeg = []
        for steer_angle in list_of_steer_angles:
            stability_index_data = yaw_moment_data[
                np.round(yaw_moment_data["Steering Wheel Angle [rad]"], decimals=8) == steer_angle
            ]
            stability_index_data = (
                stability_index_data[["Yaw Moment [Nm]", "Chassis Slip Angle [deg]"]].copy(deep=True).diff(axis=0)
            )
            static_directional_stability_nmpdeg.extend(
                stability_index_data["Yaw Moment [Nm]"] / stability_index_data["Chassis Slip Angle [deg]"]
            )

        yaw_moment_data["Static Directional Stability [Nm/deg]"] = static_directional_stability_nmpdeg

    @staticmethod
    def control_force_index(yaw_moment_data: df):
        """
        Slope of lateral acceleration at a given chassis slip angle (dAy/ddelta)
        Indicates turn-in capability of the vehicle
        """
        yaw_moment_data.sort_values(["Chassis Slip Angle [rad]", "Steering Wheel Angle [rad]"], inplace=True)
        list_of_chassis_slip_angles = np.round(yaw_moment_data["Chassis Slip Angle [rad]"], decimals=8).unique()

        control_index_gpdeg = []
        for chassis_slip_angle in list_of_chassis_slip_angles:
            control_index_data = yaw_moment_data[
                np.round(yaw_moment_data["Chassis Slip Angle [rad]"], decimals=8) == chassis_slip_angle
            ]
            control_index_data = (
                control_index_data[["Normalized Lateral Acceleration [g]", "Steering Wheel Angle [deg]"]]
                .copy(deep=True)
                .diff(axis=0)
            )
            control_index_gpdeg.extend(
                control_index_data["Normalized Lateral Acceleration [g]"]
                / control_index_data["Steering Wheel Angle [deg]"]
            )

        yaw_moment_data["Control Force Index [g/deg]"] = control_index_gpdeg

    @staticmethod
    def control_moment_index(yaw_moment_data: df):
        """
        Slope of yaw moment at a given chassis slip angle (dN/ddelta)
        Indicates turn-in capability of the vehicle
        """
        yaw_moment_data.sort_values(["Chassis Slip Angle [rad]", "Steering Wheel Angle [rad]"], inplace=True)
        list_of_chassis_slip_angles = np.round(yaw_moment_data["Chassis Slip Angle [rad]"], decimals=8).unique()

        control_index_nmpdeg = []
        for chassis_slip_angle in list_of_chassis_slip_angles:
            control_index_data = yaw_moment_data[
                np.round(yaw_moment_data["Chassis Slip Angle [rad]"], decimals=8) == chassis_slip_angle
            ]
            control_index_data = (
                control_index_data[["Yaw Moment [Nm]", "Steering Wheel Angle [deg]"]].copy(deep=True).diff(axis=0)
            )
            control_index_nmpdeg.extend(
                control_index_data["Yaw Moment [Nm]"] / control_index_data["Steering Wheel Angle [deg]"]
            )

        yaw_moment_data["Control Moment Index [Nm/deg]"] = control_index_nmpdeg

    @staticmethod
    def yaw_stability_index(yaw_moment_data: df):
        """
        Slope of yaw moment curve (dN/dAy) for constant steer
        """
        yaw_moment_data.sort_values(["Steering Wheel Angle [rad]", "Chassis Slip Angle [rad]"], inplace=True)
        list_of_steer_angles = np.round(yaw_moment_data["Steering Wheel Angle [rad]"], decimals=8).unique()

        stability_index = []
        for steer_angle in list_of_steer_angles:
            stability_index_data = yaw_moment_data[
                np.round(yaw_moment_data["Steering Wheel Angle [rad]"], decimals=8) == steer_angle
            ]
            stability_index_data = (
                stability_index_data[["Normalized Yaw Moment [-]", "Normalized Lateral Acceleration [g]"]]
                .copy(deep=True)
                .diff(axis=0)
            )
            stability_index.extend(
                stability_index_data["Normalized Yaw Moment [-]"]
                / stability_index_data["Normalized Lateral Acceleration [g]"]
            )

        yaw_moment_data["Stability Index [-]"] = stability_index

    @staticmethod
    def plot_yaw_moment_diagram(yaw_moment_data: df, speed_mps: float, in_si_units: bool = False):

        lateral_accel = "Lateral Acceleration [m/s^2]" if in_si_units else "Normalized Lateral Acceleration [g]"
        yaw_moment = "Yaw Moment [Nm]" if in_si_units else "Normalized Yaw Moment [-]"
        angle_decimals = ".2f" if in_si_units else ".0f"
        angle_units = "rad" if in_si_units else "°"
        speed_units = "m/s" if in_si_units else "km/hr"

        figure, axes = plt.subplots(nrows=1, ncols=1)
        yaw_moment_data.sort_values(["Chassis Slip Angle [rad]", "Steering Wheel Angle [rad]"], inplace=True)
        chassis_slip_angles_rad = np.round(yaw_moment_data["Chassis Slip Angle [rad]"], decimals=4).unique()
        steering_wheel_angles_rad = np.round(yaw_moment_data["Steering Wheel Angle [rad]"], decimals=4).unique()

        for chassis_slip in chassis_slip_angles_rad:
            const_slip_line_data = yaw_moment_data[
                np.round(yaw_moment_data["Chassis Slip Angle [rad]"], decimals=4) == chassis_slip
            ]
            const_slip_line_data = const_slip_line_data.sort_values("Steering Wheel Angle [rad]")
            (line,) = axes.plot(
                const_slip_line_data[lateral_accel], const_slip_line_data[yaw_moment], color="b", linewidth=0.5
            )
            if chassis_slip == chassis_slip_angles_rad[-1]:
                chassis_slip = np.rad2deg(chassis_slip) if not in_si_units else chassis_slip
                line.set_label(f"Constant Chassis Slip Angle (±{chassis_slip: {angle_decimals}}{angle_units})")

        for steer_angle in steering_wheel_angles_rad:
            const_steer_line_data = yaw_moment_data[
                np.round(yaw_moment_data["Steering Wheel Angle [rad]"], decimals=4) == steer_angle
            ]
            const_steer_line_data = const_steer_line_data.sort_values("Chassis Slip Angle [rad]")
            (line,) = axes.plot(
                const_steer_line_data[lateral_accel], const_steer_line_data[yaw_moment], color="r", linewidth=0.5
            )
            if steer_angle == steering_wheel_angles_rad[-1]:
                steer_angle = np.rad2deg(steer_angle) if not in_si_units else steer_angle
                line.set_label(f"Constant Steering Wheel Angle (±{steer_angle: {angle_decimals}}{angle_units})")

        title_speed = speed_mps if in_si_units else UnitConversion.mps_to_kph(speed_mps)
        plt.title(f"Yaw Moment Diagram at {title_speed: .0f} {speed_units}")
        plt.xlabel("Lateral Acceleration [m/s^2]") if in_si_units else plt.xlabel("Lateral Acceleration [g]")
        plt.ylabel("Yaw Moment [Nm]") if in_si_units else plt.ylabel("Yaw Moment [-]")
        plt.minorticks_on()
        plt.grid(which="minor", linewidth=0.2)
        plt.grid(which="major", linewidth=0.4)
        plt.legend(loc="upper right")
        plt.autoscale()
        mng = plt.get_current_fig_manager()
        mng.set_window_title("Yaw Moment Diagram")
        plt.show()


def main():
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
    ymd.calculate_performance_indexes(yaw_moment_data=ymd_data)
    # ymd.plot_yaw_moment_diagram(yaw_moment_data=ymd_data, speed_mps=ymd.speed_mps, in_si_units=True)
    ymd.plot_yaw_moment_diagram(yaw_moment_data=ymd_data, speed_mps=ymd.speed_mps, in_si_units=False)
    print("hi")


if __name__ == "__main__":
    main()
