from dataclasses import dataclass

from numpy import deg2rad
from lib.UnitConversions import UnitConversion


@dataclass
class VehicleSweepLimits:
    """
    Comment the parameters you don't want to sweep and change sweep limits as
    desired for running yaw_momenty sweeps
    """
    # Mass Parameters
    mass_car_kg = [UnitConversion.lb_to_kg(400), UnitConversion.lb_to_kg(650)]
    # mass_driver_kg = [40, 100]
    # mass_unsprung_f_kg = [6, 16]
    cg_height_total_m = [UnitConversion.in_to_m(6), UnitConversion.in_to_m(18)]
    mass_distribution_f = [.3, .5]

    # Dimension Attributes
    track_f_m = [UnitConversion.in_to_m(42), UnitConversion.in_to_m(52)]
    track_r_m = [UnitConversion.in_to_m(42), UnitConversion.in_to_m(52)]
    wheelbase_m = [1.525, UnitConversion.in_to_m(70)]  # comp rules min = 1.525
    roll_center_height_f_m = [0.01, 0.04]
    roll_center_height_r_m = [0.01, 0.04]

    spring_f_npm = [UnitConversion.lbpin_to_npm(200), UnitConversion.lbpin_to_npm(550)]
    spring_r_npm = [UnitConversion.lbpin_to_npm(200), UnitConversion.lbpin_to_npm(550)]
    motion_ratio_f = [0.8, 1.2]
    motion_ratio_r = [0.8, 1.2]
    arb_f_nmpdeg = [0, 600]
    arb_r_nmpdeg = [0, 600]
    # motion_ratio_arb_f = [1, 1]
    # motion_ratio_arb_r = [1, 1]
    static_incln_fl_rad = [deg2rad(-3), deg2rad(3)]
    static_incln_rl_rad = [deg2rad(-3), deg2rad(3)]
    static_toe_fl_rad = [deg2rad(-3), deg2rad(3)]
    static_toe_rl_rad = [deg2rad(-3), deg2rad(3)]

    tire_pressure_f_pa = [UnitConversion.psi_to_pa(8), UnitConversion.psi_to_pa(14)]
    tire_pressure_r_pa = [UnitConversion.psi_to_pa(8), UnitConversion.psi_to_pa(14)]

    aero_installed = [False, True]
    aero_balance_f = [0.4, 0.6]
    coeff_of_lift = [3, 4]
