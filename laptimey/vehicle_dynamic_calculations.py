from vehicle_model import VehicleParameters

def longitudinal_load_transfer_calculation(
   long_accel_mps2: float, load_transfer_drag_r_n: float, car: VehicleParameters
) -> float:
    """Returns the change in load for individual tires in the front and rear
    longitudinal force is positive in acceleration, negative in deceleration
    - Ignore load transfer from pitch angle
    - Treats body as single point mass
    """
    load_transfer_r_n = (car.cg_height_total_m * long_accel_mps2) / car.cg_to_r_wheels_dist_m + load_transfer_drag_r_n
    return load_transfer_r_n / 2


def lateral_load_transfer_left_calculation(
    lat_accel_mps2: float, car: VehicleParameters
):
    """'
    Assumptions:
    - Load transfer as a result of chassis roll/sprung mass cg change is ignored (small angles assumed)
    - Front and rear roll rates are measured independently
    - Chassis is infinitely stiff
    - Tire deflection rates are included in roll rates
    - Car is symmetric across it's longitudinal plane
    - lateral acceleration is lateral relative to the coordinate frame of the chassis
    """

    load_transfer_f_n = lat_accel_mps2 * car.lat_load_transfer_sensitivity_f_ns2pm
    load_transfer_r_n = lat_accel_mps2 * car.lat_load_transfer_sensitivity_r_ns2pm

    return load_transfer_f_n, load_transfer_r_n


def roll_angle_calculation(lateral_accel_units: float, roll_gradient_units: float) -> float:
    pass


def pitch_angle_calculation(longitudinal_accel_units: float, pitch_gradient_unts: float) -> float:
    pass


def aero_forces_calculation(car: VehicleParameters):
    aero_drag_n = 0
    load_transfer_drag_r_n = (car.center_of_pressure_height_m * aero_drag_n) / car.cg_to_r_wheels_dist_m


if __name__ == "__main__":
    print('hi')
