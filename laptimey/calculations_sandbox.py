"""
Bit of a playground for me to start getting calculations down for future use
once I figure out how to lay everything out and what I'm actually doing

Acronyms
f = front
r = rear
cg = center of gravity
moi = mass moment of inertia
n = Newtons
deg = degrees
npm = newtons per meter
npdeg = newtons per degree
kgm2= kg * m^2
nmpdeg = N * m / deg
arb = anti-roll bar
long= longitudinal
lat = lateral
accel = acceleration
motion ratio = delta suspension / delta wheel


Created 2021, Contributors: Nigel Swab
"""

from dataclasses import dataclass
import math


def main():
    pass


# @dataclass
# class VehicleParams:
#
#     def __init__(self, general, suspension, aero, powertrain, drivetrain, tires):
#         self.general = general
#         self.suspension = suspension
#         self.aero = aero
#         self.powertrain = powertrain
#         self.drivetrain = drivetrain
#         self.tires = tires

@dataclass
class VehicleParams:
    __slots__ = ['spring_f_npm',
                 'spring_r_npm',
                 'arb_f_nmpdeg',
                 'arb_r_nmpdeg',
                 'motion_ratio_f',
                 'motion_ratio_r',
                 'camber_f_deg',
                 'camber_r_deg',
                 'toe_f_deg',
                 'toe_r_deg',
                 'roll_center_height_f_m',
                 'roll_center_height_r_m',
                 'wheelrate_f_npm',
                 'wheelrate_r_npm',
                 'ride_rate_f_npm',
                 'ride_rate_r_npm',
                 'roll_stiffness_f_nmpdeg',
                 'roll_stiffness_r_nmpdeg',
                 'roll_center_to_cg_vert_dist_f_m',
                 'roll_center_to_cg_vert_dist_r_m',
                 'roll_stiffness_distribution_f',
                 'roll_stiffness_distribution_r',
                 'track_f_m',
                 'track_r_m',
                 'wheelbase_m',
                 'mass_total_kg',
                 'mass_sprung_kg',
                 'mass_unsprung_f_kg',
                 'mass_unsprung_r_kg',
                 'sprung_mass_distribution_f',
                 'sprung_mass_distribution_r',
                 'unsprung_mass_distribution_f',
                 'unsprung_mass_distribution_r',
                 'mass_driver_kg',
                 'weight_total_n',
                 'weight_distribution_f',
                 'weight_distribution_r',
                 'static_wheel_load_r_n',
                 'static_wheel_load_f_n',
                 'cg_height_total_m',
                 'cg_height_unsprung_f_m',
                 'cg_height_unsprung_r_m',
                 'cg_height_sprung_m',
                 'cg_to_f_wheels_dist_m',
                 'cg_to_r_wheels_dist_m',
                 'moi_roll_kgm2',
                 'moi_yaw_kgm2',
                 'moi_pitch_kgm2',
                 'tire_spring_f_npm',
                 'tire_spring_r_npm',
                 'tire_rad_f_m',
                 'tire_rad_r_m',
                 'tire_rad_loaded_f_m',
                 'tire_rad_loaded_r_m',
                 'aero_installed',
                 'air_density_kgm3',
                 'coeff_of_lift',
                 'coeff_of_drag',
                 'center_of_pressure_height_m',
                 'lift_distribution_long',
                 'frontal_area_m2',
                 'tire_spring_rate_f_npm',
                 'tire_spring_rate_r_npm',
                 'tire_pressure_f_pa',
                 'tire_pressure_r_pa']


def calculate_suspension_parameters(car: VehicleParams) -> None:
    car.static_wheel_load_f_n = (car.weight_total_n * car.weight_distribution_f) / 2
    car.static_wheel_load_r_n = (car.weight_total_n * car.weight_distribution_r) / 2
    car.cg_to_f_wheels_dist_m = car.wheelbase_m * car.weight_distribution_f
    car.cg_to_r_wheels_dist_m = car.wheelbase_m * car.weight_distribution_r
    car.wheelrate_f_npm = car.spring_f_npm * car.motion_ratio_f ** 2
    car.wheelrate_r_npm = car.spring_r_npm * car.motion_ratio_r ** 2
    car.ride_rate_f_npm = (car.wheelrate_f_npm ** -1 + car.tire_spring_rate_f_npm ** -1) ** -1
    car.ride_rate_r_npm = (car.wheelrate_r_npm ** -1 + car.tire_spring_rate_r_npm ** -1) ** -1
    car.roll_center_to_cg_vert_dist_f_m = car.cg_height_total_m - car.roll_center_height_f_m
    car.roll_center_to_cg_vert_dist_r_m = car.cg_height_total_m - car.roll_center_height_r_m

    car.roll_stiffness_f_nmpdeg, car.roll_stiffness_r_nmpdeg = roll_stiffness_calculation(car)
    car.roll_stiffness_distribution_f = car.roll_stiffness_f_nmpdeg / \
                                        (car.roll_stiffness_f_nmpdeg + car.roll_stiffness_r_nmpdeg)
    car.roll_stiffness_distribution_r = 1 - car.roll_stiffness_distribution_f


def sprung_cg_determination(car: VehicleParams):
    car.mass_sprung_kg = car.mass_total_kg - 2 * car.mass_unsprung_f_kg + 2 * car.mass_unsprung_r_kg
    car.unsprung_mass_distribution_f = car.mass_unsprung_f_kg / (car.mass_unsprung_f_kg + car.mass_unsprung_r_kg)
    car.unsprung_mass_distribution_r = 1 - car.unsprung_mass_distribution_f

    car.sprung_mass_distribution_f = (car.weight_distribution_f -
                                      ((car.mass_unsprung_f_kg * 2 + car.mass_unsprung_r_kg * 2)
                                       / car.mass_total_kg) * car.unsprung_mass_distribution_f) \
                                    * car.mass_total_kg / car.mass_sprung_kg
    car.sprung_mass_distribution_r = 1 - car.sprung_mass_distribution_f

    # center of unsprung mass assumed to be at tire
    car.cg_height_sprung_m = car.mass_total_kg / car.mass_sprung_kg * car.cg_height_total_m \
                             - 2 * car.mass_unsprung_f_kg / car.mass_sprung_kg * car.tire_rad_f_m \
                             - 2 * car.mass_unsprung_r_kg / car.mass_sprung_kg * car.tire_rad_r_m


def roll_stiffness_calculation(car: VehicleParams) -> tuple[float, float]:
    ''' calculates the spring rates of the wheel and tire with respect '''

    # math.tan(math.radians(1)) is used to convert into terms of 1 degree of body roll
    spring_roll_stiffness_f_nmpdeg = (car.track_f_m ** 2 * math.tan(math.radians(1)) * car.wheelrate_f_npm) / 2
    spring_roll_stiffness_r_nmpdeg = (car.track_r_m ** 2 * math.tan(math.radians(1)) * car.wheelrate_r_npm) / 2

    # math.tan(math.radians(1)) is used to convert into terms of 1 degree of body roll
    tire_roll_stiffness_f_nmpdeg = (car.track_f_m ** 2 * math.tan(math.radians(1)) * car.tire_spring_rate_f_npm) / 2
    tire_roll_stiffness_r_nmpdeg = (car.track_r_m ** 2 * math.tan(math.radians(1)) * car.tire_spring_rate_r_npm) / 2

    # Assumes that ARB rate is defined as Nm/deg of body roll
    spring_and_arb_roll_stiffness_f_nmpdeg = spring_roll_stiffness_f_nmpdeg + car.arb_f_nmpdeg
    spring_and_arb_roll_stiffness_r_nmpdeg = spring_roll_stiffness_r_nmpdeg + car.arb_r_nmpdeg

    # Equivalent spring rate for springs in series
    total_roll_stiffness_f_nmpdeg = (spring_and_arb_roll_stiffness_f_nmpdeg * tire_roll_stiffness_f_nmpdeg) /\
                                    (spring_and_arb_roll_stiffness_f_nmpdeg + tire_roll_stiffness_f_nmpdeg)
    total_roll_stiffness_r_nmpdeg = (spring_and_arb_roll_stiffness_r_nmpdeg * tire_roll_stiffness_r_nmpdeg) /\
                                    (spring_and_arb_roll_stiffness_r_nmpdeg + tire_roll_stiffness_r_nmpdeg)

    return total_roll_stiffness_f_nmpdeg, total_roll_stiffness_r_nmpdeg


def longitudinal_load_transfer_calculation(car: VehicleParams,
                                           long_accel_n: float,
                                           aero_drag_n: float) -> tuple[float, float]:
    '''Returns the change in load for individual tires in the front and rear'''
    load_change_drag_r_n = (car.center_of_pressure_height_m * aero_drag_n) / car.cg_to_r_wheels_dist_m
    load_change_r_n = ((car.cg_height_total_m * long_accel_n) / car.cg_to_r_wheels_dist_m + load_change_drag_r_n) / 2
    load_change_f_n = -load_change_r_n
    return load_change_f_n, load_change_r_n


def lateral_load_transfer_calculation(lat_accel_n: float, long_accel: float, car: VehicleParams):
    # delta_f_n = lat_accel_n * (car.weight_total_n/car.track_f_m) *\
    #             ((car.roll_center_to_cg_vert_dist_f_m * (roll)))
    pass


def aero_forces_calculation(car: VehicleParams):
    pass


if __name__ == '__main__':
    main()
