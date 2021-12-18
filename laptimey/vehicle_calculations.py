from vehicle_model import VehicleParameters


def longitudinal_load_transfer_calculation(
   long_accel_mps2: float, load_transfer_drag_r_n: float, cg_height_total_m: float, cg_to_r_wheels_dist_m: float
) -> float:
    """Returns the change in load for individual tires in the front and rear
    longitudinal force is positive in acceleration, negative in deceleration
    - Ignore load transfer from pitch angle
    - Treats body as single point mass
    """
    load_transfer_r_n = (cg_height_total_m * long_accel_mps2) / cg_to_r_wheels_dist_m + load_transfer_drag_r_n
    return load_transfer_r_n / 2


def lateral_load_transfer_left_calculation(lat_accel_mps2: float, car: VehicleParameters) -> tuple[float, float]:
    """'
    Assumptions:
    - Load transfer as a result of chassis roll/sprung mass cg change is ignored (small angles assumed)
    - Front and rear roll rates are measured independently
    - Chassis is infinitely stiff
    - Tire deflection rates are included in roll rates
    - Car is symmetric across it's longitudinal plane
    - lateral acceleration is lateral relative to the coordinate frame of the chassis
    """

    # lateral acceleration negative as forces acting on cg is in opposite direction
    load_transfer_f_n = -lat_accel_mps2 * car.lat_load_transfer_sensitivity_f_ns2pm
    load_transfer_r_n = -lat_accel_mps2 * car.lat_load_transfer_sensitivity_r_ns2pm

    return load_transfer_f_n, load_transfer_r_n


def wheel_load_check(wheel_load_l_n: float, wheel_load_r_n: float) -> tuple[float, float]:
    if wheel_load_l_n < 0:
        # wheel_load_r_n -= wheel_load_l_n
        wheel_load_l_n = 0
    if wheel_load_r_n < 0:
        # wheel_load_l_n -= wheel_load_r_n
        wheel_load_r_n = 0
    return wheel_load_l_n, wheel_load_r_n


def roll_angle_calculation(lateral_accel_mps2: float, roll_gradient_radpg: float) -> float:
    return lateral_accel_mps2 / 9.81 * roll_gradient_radpg


def pitch_angle_calculation(longitudinal_accel_units: float, pitch_gradient_unts: float) -> float:
    pass


def aerodynamic_wheel_loads_n(speed_mps: float, car: VehicleParameters) -> tuple[float, float]:
    if not car.aero_installed:
        return 0, 0

    # calculate aerodynamic forces
    downforce_total_n = aerodynamic_downforce_n(speed_mps=speed_mps, car=car)
    aero_drag_n = aerodynamic_drag_n(speed_mps=speed_mps, car=car)

    # calculate the load on a single rear wheel as a result of drag forces
    drag_wheel_load_r_n = (car.center_of_pressure_height_m * aero_drag_n) / car.cg_to_r_wheels_dist_m / 2

    # calculate the load on a single wheel due to aerodynamic forces
    downforce_wheel_load_f_n = downforce_total_n * car.aero_balance_f / 2 - drag_wheel_load_r_n
    downforce_wheel_load_r_n = downforce_total_n * (1 - car.aero_balance_f) / 2 + drag_wheel_load_r_n

    return downforce_wheel_load_f_n, downforce_wheel_load_r_n


def aerodynamic_downforce_n(speed_mps: float, car: VehicleParameters) -> float:
    return car.coeff_of_lift / 2 * car.air_density_kgm3 * speed_mps ** 2 * car.frontal_area_m2


def aerodynamic_drag_n(speed_mps: float, car: VehicleParameters) -> float:
    return car.coeff_of_drag / 2 * car.air_density_kgm3 * speed_mps ** 2 * car.frontal_area_m2


def suspension_bump_m(wheel_load_n: float, wheelrate_npm: float) -> float:
    return wheel_load_n / wheelrate_npm


def roll_center_height_m(roll_center_gain: float, bump_m: float, static_roll_center_height_m: float) -> float:
    return roll_center_gain * bump_m + static_roll_center_height_m


if __name__ == "__main__":
    car = VehicleParameters()
    transfer_f_n, transfer_r_n = lateral_load_transfer_left_calculation(lat_accel_mps2=2*9.81, car=car)
    print(f'transfer front = {transfer_f_n} N \n'
          f'transfer rear = {transfer_r_n} N')
