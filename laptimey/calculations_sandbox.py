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

All angles are in rads, or converted using numpy.deg2rad

Created 2021, Contributors: Nigel Swab
"""

from dataclasses import dataclass
import numpy as np

from lib import pyqt_helper
from tires import Mf61Coefficients


class UnitConversion:
    @staticmethod
    def psi_to_pa(psi: float) -> float:
        return psi * 6894.76

    @staticmethod
    def pa_to_psi(psi: float) -> float:
        return psi / 6894.76

    @staticmethod
    def mps2_to_g(mps2: float) -> float:
        return mps2 / 9.80665

    @staticmethod
    def g_to_mps2(g: float) -> float:
        return g * 9.80665

    @staticmethod
    def lbpin_to_npm(lbpin: float) -> float:
        return lbpin * 175.126835

    @staticmethod
    def npm_to_lbpin(npm: float) -> float:
        return npm / 175.126835

    @staticmethod
    def in_to_m(inches: float) -> float:
        return inches * 0.0254

    @staticmethod
    def m_to_in(m: float) -> float:
        return m / 0.0254

    @staticmethod
    def lb_to_kg(lb: float) -> float:
        return lb * 0.453592

    @staticmethod
    def kg_to_lb(kg: float) -> float:
        return kg / 0.453592


@dataclass
class VehicleParams:
    __slots__ = ['spring_f_npm',
                 'spring_r_npm',
                 'arb_f_nmpdeg',
                 'arb_r_nmpdeg',
                 'motion_ratio_f',
                 'motion_ratio_r',
                 'motion_ratio_arb_f',
                 'motion_ratio_arb_r',
                 'camber_f_rad',
                 'camber_r_rad',
                 'toe_f_rad',
                 'toe_r_rad',
                 'roll_center_height_f_m',
                 'roll_center_height_r_m',
                 'wheelrate_f_npm',
                 'wheelrate_r_npm',
                 'ride_rate_f_npm',
                 'ride_rate_r_npm',
                 'roll_stiffness_f_nmpdeg',
                 'roll_stiffness_r_nmpdeg',
                 'roll_stiffness_f_nmprad',
                 'roll_stiffness_r_nmprad',
                 'roll_stiffness_distribution_f',
                 'roll_stiffness_distribution_r',
                 'roll_axis_to_cg_dist_vert_m',
                 'roll_axis_to_cg_dist_perp_m',
                 'roll_gradient_degpg',
                 'track_f_m',
                 'track_r_m',
                 'wheelbase_m',
                 'rideheight_f_m',
                 'rideheight_r_m',
                 'mass_car_kg',
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
                 'weight_sprung_n',
                 'mass_distribution_f',
                 'mass_distribution_r',
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
                 'tire_pressure_f_pa',
                 'tire_pressure_r_pa',
                 'lat_load_transfer_sensitivity_f_ns2pm',
                 'lat_load_transfer_sensitivity_r_ns2pm']


def load_vehicle_from_yaml() -> VehicleParams:
    # For testing purposes, create configy instance -> config will eventually be passed in
    qt_helper = pyqt_helper.Dialogs(__file__ + 'get yaml')

    # Load configuration
    config, _ = qt_helper.load_yamls()
    config = config[0]

    # Create instance of vehicle model
    car = VehicleParams()

    # mass properties
    car.mass_driver_kg = UnitConversion.lb_to_kg(config['mass_driver_lb'])
    car.mass_car_kg = UnitConversion.lb_to_kg(config['mass_car_lb'])
    car.mass_unsprung_f_kg = UnitConversion.lb_to_kg(config['mass_unsprung_f_lb'])
    car.mass_unsprung_r_kg = UnitConversion.lb_to_kg(config['mass_unsprung_r_lb'])
    car.mass_distribution_f = config['mass_distribution_f']
    car.cg_height_total_m = config['cg_height_total_m']
    car.moi_roll_kgm2 = config['moi_roll_kgm2']
    car.moi_pitch_kgm2 = config['moi_pitch_kgm2']
    car.moi_yaw_kgm2 = config['moi_yaw_kgm2']
    # TODO: Would be nice to have some way of accounting for driver mass on cg location

    # Vehicle Dimensions
    car.track_f_m = UnitConversion.in_to_m(config['track_f_in'])
    car.track_r_m = UnitConversion.in_to_m(config['track_r_in'])
    car.wheelbase_m = UnitConversion.in_to_m(config['wheelbase_in'])
    car.rideheight_f_m = config['rideheight_f_m']
    car.rideheight_r_m = config['rideheight_r_m']
    car.roll_center_height_f_m = config['roll_center_height_f_m']
    car.roll_center_height_r_m = config['roll_center_height_r_m']
    # TODO: add variables to correlate change in rideheight with change in roll centers

    # Tire Physical Parameters
    car.tire_rad_f_m = UnitConversion.in_to_m(config['tire_rad_f_in'])
    car.tire_rad_r_m = UnitConversion.in_to_m(config['tire_rad_r_in'])
    car.tire_spring_f_npm = UnitConversion.lbpin_to_npm(config['tire_spring_rate_f_lbpin'])
    car.tire_spring_r_npm = UnitConversion.lbpin_to_npm(config['tire_spring_rate_r_lbpin'])
    car.tire_pressure_f_pa = UnitConversion.psi_to_pa(config['tire_pressure_f_psi'])
    car.tire_pressure_r_pa = UnitConversion.psi_to_pa(config['tire_pressure_r_psi'])
    # TODO: Add spring rate model to correlate pressure to spring rate

    # Suspension Setup Parameters
    car.spring_f_npm = UnitConversion.lbpin_to_npm(config['spring_f_lbpin'])
    car.spring_r_npm = UnitConversion.lbpin_to_npm(config['spring_r_lbpin'])
    car.motion_ratio_f = config['motion_ratio_f']
    car.motion_ratio_r = config['motion_ratio_r']
    car.arb_f_nmpdeg = config['arb_f_nmpdeg']
    car.arb_r_nmpdeg = config['arb_r_nmpdeg']
    car.motion_ratio_arb_f = config['motion_ratio_arb_f']
    car.motion_ratio_arb_r = config['motion_ratio_arb_r']
    car.camber_f_rad = np.deg2rad(config['camber_f_deg'])
    car.camber_r_rad = np.deg2rad(config['camber_r_deg'])
    car.toe_f_rad = np.deg2rad(config['toe_f_deg'])
    car.toe_r_rad = np.deg2rad(config['toe_r_deg'])

    # Aero Parameters
    car.aero_installed = config['aero_installed']
    car.air_density_kgm3 = config['air_density_kgm3']
    car.coeff_of_lift = config['coeff_of_lift']
    car.coeff_of_drag = config['coeff_of_drag']
    car.center_of_pressure_height_m = config['center_of_pressure_height_m']
    car.lift_distribution_long = config['lift_distribution_long']
    car.frontal_area_m2 = config['frontal_area_m2']

    # calculated values
    mass_distribution_definitions(car=car)
    parameters_definition(car=car)
    car.tire_rad_loaded_f_m, car.tire_rad_loaded_r_m = tire_radius(car, car.static_wheel_load_f_n,
                                                                   car.static_wheel_load_r_n)
    cg_definitions(car=car)
    roll_axis_dimensions(car=car)
    roll_stiffness_calculation(car=car)
    roll_sensitivity_definition(car=car)
    return car


def parameters_definition(car: VehicleParams) -> None:

    car.wheelrate_f_npm = car.spring_f_npm * car.motion_ratio_f ** 2
    car.wheelrate_r_npm = car.spring_r_npm * car.motion_ratio_r ** 2
    car.ride_rate_f_npm = (car.wheelrate_f_npm ** -1 + car.tire_spring_f_npm ** -1) ** -1
    car.ride_rate_r_npm = (car.wheelrate_r_npm ** -1 + car.tire_spring_r_npm ** -1) ** -1

    car.roll_stiffness_f_nmpdeg, car.roll_stiffness_r_nmpdeg = roll_stiffness_calculation(car)
    car.roll_stiffness_f_nmprad = np.rad2deg(car.roll_stiffness_f_nmpdeg)
    car.roll_stiffness_r_nmprad = np.rad2deg(car.roll_stiffness_r_nmpdeg)
    car.roll_stiffness_distribution_f = car.roll_stiffness_f_nmpdeg \
        / (car.roll_stiffness_f_nmpdeg + car.roll_stiffness_r_nmpdeg)
    car.roll_stiffness_distribution_r = 1 - car.roll_stiffness_distribution_f


def mass_distribution_definitions(car: VehicleParams) -> None:
    # Mass definitions
    car.mass_total_kg = car.mass_car_kg + car.mass_driver_kg
    car.weight_total_n = car.mass_total_kg * 9.81
    car.mass_sprung_kg = car.mass_total_kg - 2 * car.mass_unsprung_f_kg + 2 * car.mass_unsprung_r_kg
    car.weight_sprung_n = car.mass_sprung_kg * 9.81


    # Rear mass distribution
    car.mass_distribution_r = 1 - car.mass_distribution_f

    # Unsprung mass distribution
    car.unsprung_mass_distribution_f = car.mass_unsprung_f_kg / (car.mass_unsprung_f_kg + car.mass_unsprung_r_kg)
    car.unsprung_mass_distribution_r = 1 - car.unsprung_mass_distribution_f

    # Sprung mass distribution
    car.sprung_mass_distribution_f = (car.mass_distribution_f
                                      - ((car.mass_unsprung_f_kg * 2 + car.mass_unsprung_r_kg * 2)
                                         / car.mass_total_kg) * car.unsprung_mass_distribution_f) \
                                     * car.mass_total_kg / car.mass_sprung_kg
    car.sprung_mass_distribution_r = 1 - car.sprung_mass_distribution_f

    # Static wheel loads: Assumes symmetry
    car.static_wheel_load_f_n = (car.weight_total_n * car.mass_distribution_f) / 2
    car.static_wheel_load_r_n = (car.weight_total_n * car.mass_distribution_r) / 2

    # Length from cg to each axle
    car.cg_to_f_wheels_dist_m = car.wheelbase_m * car.mass_distribution_r
    car.cg_to_r_wheels_dist_m = car.wheelbase_m * car.mass_distribution_f


def cg_definitions(car: VehicleParams) -> None:
    # Unspring CG assumed to be at center of wheel (unloaded)
    car.cg_height_unsprung_f_m = car.tire_rad_f_m
    car.cg_height_unsprung_r_m = car.tire_rad_r_m

    # Sprung mass CG height
    car.cg_height_sprung_m = car.mass_total_kg / car.mass_sprung_kg * car.cg_height_total_m \
        - 2 * car.mass_unsprung_f_kg / car.mass_sprung_kg * car.cg_height_unsprung_f_m \
        - 2 * car.mass_unsprung_r_kg / car.mass_sprung_kg * car.cg_height_unsprung_r_m


def tire_radius(car: VehicleParams, wheel_load_f_n: float, wheel_load_r_n: float) -> tuple[float, float]:
    tire_rad_f_m = car.tire_rad_f_m - (wheel_load_f_n / car.tire_spring_f_npm)
    tire_rad_r_m = car.tire_rad_r_m - (wheel_load_r_n / car.tire_spring_r_npm)
    return tire_rad_f_m, tire_rad_r_m


def roll_stiffness_calculation(car: VehicleParams) -> tuple[float, float]:
    ''' calculates the spring rates of the wheel and tire with respect '''
    # Convert ARB rate to account for motion ratio
    car.arb_f_nmpdeg = car.arb_f_nmpdeg * car.motion_ratio_arb_f ** 2
    car.arb_r_nmpdeg = car.arb_r_nmpdeg * car.motion_ratio_arb_r ** 2


    # math.tan(math.radians(1)) is used to convert into terms of 1 degree of body roll
    spring_roll_stiffness_f_nmpdeg = (car.track_f_m ** 2 * np.tan(np.deg2rad(1)) * car.wheelrate_f_npm) / 2
    spring_roll_stiffness_r_nmpdeg = (car.track_r_m ** 2 * np.tan(np.deg2rad(1)) * car.wheelrate_r_npm) / 2

    # math.tan(math.radians(1)) is used to convert into terms of 1 degree of body roll
    tire_roll_stiffness_f_nmpdeg = (car.track_f_m ** 2 * np.tan(np.deg2rad(1)) * car.tire_spring_f_npm) / 2
    tire_roll_stiffness_r_nmpdeg = (car.track_r_m ** 2 * np.tan(np.deg2rad(1)) * car.tire_spring_r_npm) / 2

    # Assumes that ARB rate is defined as Nm/deg of body roll
    spring_and_arb_roll_stiffness_f_nmpdeg = spring_roll_stiffness_f_nmpdeg + car.arb_f_nmpdeg
    spring_and_arb_roll_stiffness_r_nmpdeg = spring_roll_stiffness_r_nmpdeg + car.arb_r_nmpdeg

    # Equivalent spring rate for springs in series
    total_roll_stiffness_f_nmpdeg = (spring_and_arb_roll_stiffness_f_nmpdeg * tire_roll_stiffness_f_nmpdeg) /\
                                    (spring_and_arb_roll_stiffness_f_nmpdeg + tire_roll_stiffness_f_nmpdeg)
    total_roll_stiffness_r_nmpdeg = (spring_and_arb_roll_stiffness_r_nmpdeg * tire_roll_stiffness_r_nmpdeg) /\
                                    (spring_and_arb_roll_stiffness_r_nmpdeg + tire_roll_stiffness_r_nmpdeg)

    return total_roll_stiffness_f_nmpdeg, total_roll_stiffness_r_nmpdeg


def pitch_stiffness_definition(car: VehicleParams) -> None:
    pass


def roll_sensitivity_definition(car: VehicleParams) -> None:
    '''
    Defines lateral load transfer sensitivities for the front and rear of the car in newtons seconds^2 / meter
    as are VehicleParam variable. This variable can be multiplied by lateral acceleration to calculate lateral
    load transfer to find the change in normal force on the tire at each axle

    All calculations are essentially from Pg 682 of Race Car Vehicle Dynamics

    Assumptions:
    - body roll angles are small (sin law)
    - roll centers and roll moment arms are static
    - the car is symmetric about its longitudinal plane
    '''
    # roll moment about the roll center given one 'g' of acceleration acting on the sprung mass (assumed positive)
    sprung_mass_roll_moment_nmpg = car.weight_sprung_n * car.roll_axis_to_cg_dist_perp_m

    # roll gradient in degrees of body roll per 'g' of lateral force
    roll_gradient_radpg = sprung_mass_roll_moment_nmpg \
        / (car.roll_stiffness_f_nmprad + car.roll_stiffness_r_nmprad - sprung_mass_roll_moment_nmpg)
    car.roll_gradient_degpg = np.deg2rad(roll_gradient_radpg)

    # substituting sprung cg to front/rear distribution with rear/front sprung mass distribution (respectively)
    roll_stiffness_sprung_f_nmprad = car.roll_stiffness_f_nmprad \
        - car.sprung_mass_distribution_f * sprung_mass_roll_moment_nmpg
    roll_stiffness_sprung_r_nmprad = car.roll_stiffness_r_nmprad \
        - car.sprung_mass_distribution_r * sprung_mass_roll_moment_nmpg

    # load transfer sensitivity of unsprung mass
    unsprung_load_transfer_sensitivity_f_npg = car.mass_unsprung_f_kg * 2 * 9.81 * car.cg_height_unsprung_f_m \
        / car.track_f_m
    unsprung_load_transfer_sensitivity_r_npg = car.mass_unsprung_r_kg * 2 * 9.81 * car.cg_height_unsprung_r_m \
        / car.track_r_m

    # define lateral load sensitivity in newtons of tire load change per g of lateral acceleration
    lat_load_transfer_sensitivity_f_npg = car.weight_sprung_n / car.track_f_m * \
        (roll_gradient_radpg * roll_stiffness_sprung_f_nmprad / car.weight_sprung_n +
         (car.sprung_mass_distribution_f * car.roll_center_height_f_m)) + unsprung_load_transfer_sensitivity_f_npg
    lat_load_transfer_sensitivity_r_npg = car.weight_sprung_n / car.track_r_m * \
        (roll_gradient_radpg * roll_stiffness_sprung_r_nmprad / car.weight_sprung_n +
         (car.sprung_mass_distribution_r * car.roll_center_height_r_m)) + unsprung_load_transfer_sensitivity_r_npg

    # converted to lateral load change in newtons per m/s^2 of lateral acceleration
    car.lat_load_transfer_sensitivity_f_ns2pm = lat_load_transfer_sensitivity_f_npg / 9.81
    car.lat_load_transfer_sensitivity_r_ns2pm = lat_load_transfer_sensitivity_r_npg / 9.81


def roll_axis_dimensions(car: VehicleParams):
    roll_axis_angle_rad = np.arctan((car.roll_center_height_f_m - car.roll_center_height_r_m) / car.wheelbase_m)
    car.roll_axis_to_cg_dist_vert_m = car.cg_height_total_m - car.roll_center_height_f_m \
        - (car.cg_to_f_wheels_dist_m * np.tan(roll_axis_angle_rad))

    # Derived from trig/geometric relations
    car.roll_axis_to_cg_dist_perp_m = np.cos(roll_axis_angle_rad)\
        * ((car.cg_height_sprung_m - car.roll_center_height_f_m)
            - (car.cg_to_f_wheels_dist_m * np.tan(roll_axis_angle_rad)))


def longitudinal_load_transfer_calculation(car: VehicleParams,
                                           long_accel_mps2: float,
                                           load_transfer_drag_r_n: float) -> float:
    '''Returns the change in load for individual tires in the front and rear
    longitudinal force is positive in acceleration, negative in deceleration
    - Ignore load transfer from pitch angle
    - Treats body as single point mass
    '''
    load_transfer_r_n = ((car.cg_height_total_m * long_accel_mps2) / car.cg_to_r_wheels_dist_m + load_transfer_drag_r_n)
    return load_transfer_r_n / 2


def lateral_load_transfer_calculation(lat_accel_mps2: float, car: VehicleParams):
    ''''
    Assumptions:
    - Load transfer as a result of chassis roll/sprung mass cg change is ignored (small angles assumed)
    - Front and rear roll rates are measured independently
    - Chassis is infinitely stiff
    - Tire deflection rates are included in roll rates
    - Car is symmetric across it's longitudinal plane
    - lateral acceleration is lateral relative to the coordinate frame of the chassis
    '''

    load_transfer_f_n = lat_accel_mps2 * car.lat_load_transfer_sensitivity_f_ns2pm
    load_transfer_r_n = lat_accel_mps2 * car.lat_load_transfer_sensitivity_r_ns2pm

    return load_transfer_f_n, load_transfer_r_n


def roll_angle_calculation(lateral_accel_units: float, roll_gradient_units: float) -> float:
    pass


def pitch_angle_calculation(longitudinal_accel_units: float, pitch_gradient_untis: float) -> float:
    pass


def aero_forces_calculation(car: VehicleParams):
    aero_drag_n = 0
    load_transfer_drag_r_n = (car.center_of_pressure_height_m * aero_drag_n) / car.cg_to_r_wheels_dist_m
    pass


def main():
    car = load_vehicle_from_yaml()
    lat_load_transfer_f, lat_load_transfer_r = lateral_load_transfer_calculation(2 * 9.81, car)
    print(f'\nFront Load Transfer = {lat_load_transfer_f} \n'
          f'Rear Load Transfer = {lat_load_transfer_r} \n')


if __name__ == '__main__':
    main()
