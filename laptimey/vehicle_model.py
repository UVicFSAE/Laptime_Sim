"""
This module is used to create and store all vehicle model data in classes


Created 2021, Contributors: Nigel Swab
"""
import numpy as np

from kinematic_equations import KinematicData, KinematicEquations
from lib import pyqt_helper
from lib.UnitConversions import UnitConversion
from tires import MagicFormula, Mf52, Mf61


class VehicleParameters:
    __slots__ = [
        "mass_car_kg",
        "mass_driver_kg",
        "mass_total_kg",
        "mass_unsprung_f_kg",
        "mass_unsprung_r_kg",
        "mass_distribution_f",
        "cg_height_total_m",
        "moi_pitch_kgm2",
        "moi_roll_kgm2",
        "moi_yaw_kgm2",
        "mass_total_kg",
        "weight_total_n",
        "mass_sprung_kg",
        "weight_sprung_n",
        "mass_distribution_r",
        "cg_to_f_wheels_dist_m",
        "cg_to_r_wheels_dist_m",
        "static_wheel_load_r_n",
        "static_wheel_load_f_n",
        'cg_height_unsprung_f_m',
        'cg_height_unsprung_r_m',
        'cg_height_sprung_m',
        "track_f_m",
        "track_r_m",
        "wheelbase_m",
        "rideheight_f_m",
        "rideheight_r_m",
        "spring_f_npm",
        "spring_r_npm",
        "motion_ratio_f",
        "motion_ratio_r",
        "arb_f_nmpdeg",
        "arb_r_nmpdeg",
        "motion_ratio_arb_f",
        "motion_ratio_arb_r",
        "roll_center_height_f_m",
        "roll_center_height_r_m",
        "static_camber_f_rad",
        "static_camber_r_rad",
        "static_toe_f_rad",
        "static_toe_r_rad",
        "wheelrate_f_npm",
        "wheelrate_r_npm",
        "ride_rate_f_npm",
        "ride_rate_r_npm",
        "roll_gradient_radpg",
        "kinematic_file",
        "kinematic_data",
        "kinematics_l",
        "kinematics_r",
        "lat_load_transfer_sensitivity_f_ns2pm",
        "lat_load_transfer_sensitivity_r_ns2pm",
        "roll_gradient_radpg",
        "tire_f",
        "tire_r",
        "tire_file_and_version_f",
        "tire_file_and_version_r",
        "tire_pressure_f_pa",
        "tire_pressure_r_pa",
        "tire_spring_f_npm",
        "tire_spring_r_npm",
        "tire_rad_f_m",
        "tire_rad_r_m",
        "tire_rad_loaded_f_m",
        "tire_rad_loaded_r_m",
        "aero_installed",
        "air_density_kgm3",
        "coeff_of_lift",
        "coeff_of_drag",
        "center_of_pressure_height_m",
        "lift_distribution_long",
        "frontal_area_m2",
    ]

    # INERTIAL PROPERTIES
    mass_car_kg: float
    mass_driver_kg: float
    mass_unsprung_f_kg: float
    mass_unsprung_r_kg: float
    mass_distribution_f: float
    cg_height_total_m: float
    moi_pitch_kgm2: float
    moi_roll_kgm2: float
    moi_yaw_kgm2: float
    #  calculated
    mass_total_kg: float
    weight_total_n: float  # questionable
    mass_sprung_kg: float
    weight_sprung_n: float  # questionable
    cg_height_unsprung_f_m: float  # questionable
    cg_height_unsprung_r_m: float  # questionable
    cg_height_sprung_m: float  # questionable
    cg_to_f_wheels_dist_m: float
    cg_to_r_wheels_dist_m: float
    static_wheel_load_r_n: float
    static_wheel_load_f_n: float

    # DIMENSIONS
    track_f_m: float
    track_r_m: float
    wheelbase_m: float
    rideheight_f_m: float
    rideheight_r_m: float

    # SUSPENSION PARAMETERS
    spring_f_npm: float
    spring_r_npm: float
    motion_ratio_f: float
    motion_ratio_r: float
    arb_f_nmpdeg: float
    arb_r_nmpdeg: float
    motion_ratio_arb_f: float
    motion_ratio_arb_r: float
    roll_center_height_f_m: float
    roll_center_height_r_m: float
    static_camber_f_rad: float
    static_camber_r_rad: float
    static_toe_f_rad: float
    static_toe_r_rad: float
    # setup from csv file
    kinematic_file: str
    kinematic_data: KinematicData
    kinematics_l: KinematicEquations
    kinematics_r: KinematicEquations
    # calculated
    wheelrate_f_npm: float
    wheelrate_r_npm: float
    ride_rate_f_npm: float
    ride_rate_r_npm: float
    lat_load_transfer_sensitivity_f_ns2pm: float
    lat_load_transfer_sensitivity_r_ns2pm: float
    roll_gradient_radpg: float

    # TIRE PARAMETERS
    tire_f: MagicFormula
    tire_r: MagicFormula
    tire_file_and_version_f: list[str]
    tire_file_and_version_r: list[str]
    tire_spring_f_npm: float
    tire_spring_r_npm: float
    tire_rad_f_m: float
    tire_rad_r_m: float
    tire_pressure_f_pa: float
    tire_pressure_r_pa: float

    # AERODYNAMIC PROPERTIES
    aero_installed: float
    air_density_kgm3: float
    coeff_of_lift: float
    coeff_of_drag: float
    center_of_pressure_height_m: float
    lift_distribution_long: float
    frontal_area_m2: float

    def __init__(self):
        self.load_vehicle_parameters_from_yaml()
        self.define_calculated_vehicle_parameters()
        self.tire_model_definitions()
        self.kinematic_definitions()

    def load_vehicle_parameters_from_yaml(self) -> None:
        # For testing purposes, create configy instance -> config will eventually be passed in
        qt_helper = pyqt_helper.Dialogs(__file__ + "get yaml")

        # Load configuration
        config, _ = qt_helper.load_yamls()
        config = config[0]

        # mass properties
        self.mass_driver_kg = UnitConversion.lb_to_kg(config["mass_driver_lb"])
        self.mass_car_kg = UnitConversion.lb_to_kg(config["mass_car_lb"])
        self.mass_unsprung_f_kg = UnitConversion.lb_to_kg(config["mass_unsprung_f_lb"])
        self.mass_unsprung_r_kg = UnitConversion.lb_to_kg(config["mass_unsprung_r_lb"])
        self.mass_distribution_f = config["mass_distribution_f"]
        self.cg_height_total_m = config["cg_height_total_m"]
        self.moi_roll_kgm2 = config["moi_roll_kgm2"]
        self.moi_pitch_kgm2 = config["moi_pitch_kgm2"]
        self.moi_yaw_kgm2 = config["moi_yaw_kgm2"]
        # TODO: Would be nice to have some way of accounting for driver mass on cg location

        # Vehicle Dimensions
        self.track_f_m = UnitConversion.in_to_m(config["track_f_in"])
        self.track_r_m = UnitConversion.in_to_m(config["track_r_in"])
        self.wheelbase_m = UnitConversion.in_to_m(config["wheelbase_in"])
        self.rideheight_f_m = config["rideheight_f_m"]
        self.rideheight_r_m = config["rideheight_r_m"]
        self.roll_center_height_f_m = config["roll_center_height_f_m"]
        self.roll_center_height_r_m = config["roll_center_height_r_m"]
        # TODO: add variables to correlate change in rideheight with change in roll centers

        # Tire Parameters
        self.tire_file_and_version_f = [config["tire_model_f_filepath"], config["tire_model_f_version"]]
        self.tire_file_and_version_r = [config["tire_model_r_filepath"], config["tire_model_r_version"]]
        self.tire_rad_f_m = UnitConversion.in_to_m(config["tire_rad_f_in"])
        self.tire_rad_r_m = UnitConversion.in_to_m(config["tire_rad_r_in"])
        self.tire_spring_f_npm = UnitConversion.lbpin_to_npm(config["tire_spring_rate_f_lbpin"])
        self.tire_spring_r_npm = UnitConversion.lbpin_to_npm(config["tire_spring_rate_r_lbpin"])
        self.tire_pressure_f_pa = UnitConversion.psi_to_pa(config["tire_pressure_f_psi"])
        self.tire_pressure_r_pa = UnitConversion.psi_to_pa(config["tire_pressure_r_psi"])
        # TODO: Add spring rate model to correlate pressure to spring rate

        # Suspension Setup Parameters
        self.spring_f_npm = UnitConversion.lbpin_to_npm(config["spring_f_lbpin"])
        self.spring_r_npm = UnitConversion.lbpin_to_npm(config["spring_r_lbpin"])
        self.motion_ratio_f = config["motion_ratio_f"]
        self.motion_ratio_r = config["motion_ratio_r"]
        self.arb_f_nmpdeg = config["arb_f_nmpdeg"]
        self.arb_r_nmpdeg = config["arb_r_nmpdeg"]
        self.motion_ratio_arb_f = config["motion_ratio_arb_f"]
        self.motion_ratio_arb_r = config["motion_ratio_arb_r"]
        self.static_camber_f_rad = np.deg2rad(config["static_camber_f_deg"])
        self.static_camber_r_rad = np.deg2rad(config["static_camber_r_deg"])
        self.static_toe_f_rad = np.deg2rad(config["static_toe_f_deg"])
        self.static_toe_r_rad = np.deg2rad(config["static_toe_r_deg"])
        self.kinematic_file = config["kinematic_file"] or None

        # Aero Parameters
        self.aero_installed = config["aero_installed"]
        self.air_density_kgm3 = config["air_density_kgm3"]
        self.coeff_of_lift = config["coeff_of_lift"]
        self.coeff_of_drag = config["coeff_of_drag"]
        self.center_of_pressure_height_m = config["center_of_pressure_height_m"]
        self.lift_distribution_long = config["lift_distribution_long"]
        self.frontal_area_m2 = config["frontal_area_m2"]

    def define_calculated_vehicle_parameters(self):
        self.mass_and_weight_definitions()
        self.static_wheel_load_definitions()
        self.cg_definitions()
        self.wheelrate_definitions()
        self.ride_rate_definitions()
        self.roll_gradient_definition()
        self.lateral_load_transfer_sensitivity_definitions()

    def mass_and_weight_definitions(self) -> None:
        # Mass definitions
        self.mass_total_kg = self.mass_car_kg + self.mass_driver_kg
        self.weight_total_n = self.mass_total_kg * 9.81
        self.mass_sprung_kg = self.mass_total_kg - 2 * self.mass_unsprung_f_kg + 2 * self.mass_unsprung_r_kg
        self.weight_sprung_n = self.mass_sprung_kg * 9.81

    def static_wheel_load_definitions(self) -> None:
        # Static wheel loads: Assumes symmetry
        self.static_wheel_load_f_n = (self.weight_total_n * self.mass_distribution_f) / 2
        self.static_wheel_load_r_n = (self.weight_total_n * (1 - self.mass_distribution_f)) / 2

    def cg_definitions(self) -> None:
        self.cg_unsprung_definitions()
        self.cg_sprung_definition()
        self.cg_to_wheel_definitions()

    def cg_unsprung_definitions(self) -> None:
        # Unsprung CG assumed to be at center of wheel (unloaded)
        self.cg_height_unsprung_f_m = self.tire_rad_f_m
        self.cg_height_unsprung_r_m = self.tire_rad_r_m

    def cg_sprung_definition(self) -> None:
        self.cg_height_sprung_m = (
                self.mass_total_kg / self.mass_sprung_kg * self.cg_height_total_m
                - 2 * self.mass_unsprung_f_kg / self.mass_sprung_kg * self.cg_height_unsprung_f_m
                - 2 * self.mass_unsprung_r_kg / self.mass_sprung_kg * self.cg_height_unsprung_r_m
        )

    def cg_to_wheel_definitions(self) -> None:
        # Length from cg to each axle
        self.cg_to_f_wheels_dist_m = self.wheelbase_m * (1 - self.mass_distribution_f)
        self.cg_to_r_wheels_dist_m = self.wheelbase_m * self.mass_distribution_f

    def wheelrate_definitions(self) -> None:
        self.wheelrate_f_npm = self.spring_f_npm * self.motion_ratio_f ** 2
        self.wheelrate_r_npm = self.spring_r_npm * self.motion_ratio_r ** 2

    def ride_rate_definitions(self) -> None:
        self.ride_rate_f_npm = (self.wheelrate_f_npm ** -1 + self.tire_spring_f_npm ** -1) ** -1
        self.ride_rate_r_npm = (self.wheelrate_r_npm ** -1 + self.tire_spring_r_npm ** -1) ** -1

    def calculate_sprung_mass_distribution(self) -> tuple[float, float]:
        # Unsprung mass distribution
        unsprung_mass_distribution_f = self.mass_unsprung_f_kg / (self.mass_unsprung_f_kg + self.mass_unsprung_r_kg)

        # Sprung mass distribution
        sprung_mass_distribution_f = (
                (
                        self.mass_distribution_f
                        - ((self.mass_unsprung_f_kg * 2 + self.mass_unsprung_r_kg * 2) / self.mass_total_kg)
                        * unsprung_mass_distribution_f
                )
                * self.mass_total_kg
                / self.mass_sprung_kg
        )
        sprung_mass_distribution_r = 1 - sprung_mass_distribution_f

        return sprung_mass_distribution_f, sprung_mass_distribution_r

    def calculate_roll_axis_to_cg_dist_perp_m(self) -> float:
        roll_axis_angle_rad = np.arctan((self.roll_center_height_f_m - self.roll_center_height_r_m) / self.wheelbase_m)
        return np.cos(roll_axis_angle_rad) * (
                (self.cg_height_sprung_m - self.roll_center_height_f_m)
                - (self.cg_to_f_wheels_dist_m * np.tan(roll_axis_angle_rad))
        )

    def roll_gradient_definition(self) -> None:
        # define roll axis arm length
        roll_axis_to_cg_dist_perp_m = self.calculate_roll_axis_to_cg_dist_perp_m()

        # roll moment about the roll center given one 'g' of acceleration acting on the sprung mass
        sprung_mass_roll_moment_nmpg = self.weight_sprung_n * roll_axis_to_cg_dist_perp_m

        # calculate roll stiffnesses
        roll_stiffness_f_nmprad, roll_stiffness_r_nmprad = self.calculate_roll_stiffness()

        # roll gradient in degrees of body roll per 'g' of lateral force
        self.roll_gradient_radpg = sprung_mass_roll_moment_nmpg / (
                roll_stiffness_f_nmprad + roll_stiffness_r_nmprad - sprung_mass_roll_moment_nmpg
        )

    def calculate_roll_stiffness(self) -> tuple[float, float]:
        """calculates the spring rates of the wheel and tire with respect"""
        # Convert ARB rate to account for motion ratio
        self.arb_f_nmpdeg = self.arb_f_nmpdeg * self.motion_ratio_arb_f ** 2
        self.arb_r_nmpdeg = self.arb_r_nmpdeg * self.motion_ratio_arb_r ** 2

        # math.tan(math.radians(1)) is used to convert into terms of 1 degree of body roll
        spring_roll_stiffness_f_nmpdeg = (self.track_f_m ** 2 * np.tan(np.deg2rad(1)) * self.wheelrate_f_npm) / 2
        spring_roll_stiffness_r_nmpdeg = (self.track_r_m ** 2 * np.tan(np.deg2rad(1)) * self.wheelrate_r_npm) / 2

        # math.tan(math.radians(1)) is used to convert into terms of 1 degree of body roll
        tire_roll_stiffness_f_nmpdeg = (self.track_f_m ** 2 * np.tan(np.deg2rad(1)) * self.tire_spring_f_npm) / 2
        tire_roll_stiffness_r_nmpdeg = (self.track_r_m ** 2 * np.tan(np.deg2rad(1)) * self.tire_spring_r_npm) / 2

        # Assumes that ARB rate is defined as Nm/deg of body roll
        spring_and_arb_roll_stiffness_f_nmpdeg = spring_roll_stiffness_f_nmpdeg + self.arb_f_nmpdeg
        spring_and_arb_roll_stiffness_r_nmpdeg = spring_roll_stiffness_r_nmpdeg + self.arb_r_nmpdeg

        # Equivalent spring rate for springs in series
        total_roll_stiffness_f_nmpdeg = (spring_and_arb_roll_stiffness_f_nmpdeg * tire_roll_stiffness_f_nmpdeg) / (
                spring_and_arb_roll_stiffness_f_nmpdeg + tire_roll_stiffness_f_nmpdeg
        )
        total_roll_stiffness_r_nmpdeg = (spring_and_arb_roll_stiffness_r_nmpdeg * tire_roll_stiffness_r_nmpdeg) / (
                spring_and_arb_roll_stiffness_r_nmpdeg + tire_roll_stiffness_r_nmpdeg
        )

        total_roll_stiffness_f_nmprad, total_roll_stiffness_r_nmprad = np.rad2deg(
            [total_roll_stiffness_f_nmpdeg, total_roll_stiffness_r_nmpdeg]
        )

        return total_roll_stiffness_f_nmprad, total_roll_stiffness_r_nmprad

    def calculate_roll_stiffness_sprung(
            self, sprung_mass_distribution_f: float, sprung_mass_distribution_r: float
    ) -> tuple[float, float]:
        # define roll axis arm length
        roll_axis_to_cg_dist_perp_m = self.calculate_roll_axis_to_cg_dist_perp_m()

        # roll moment about the roll center given one 'g' of acceleration acting on the sprung mass
        sprung_mass_roll_moment_nmpg = self.weight_sprung_n * roll_axis_to_cg_dist_perp_m

        # calculate roll stiffnesses
        roll_stiffness_f_nmprad, roll_stiffness_r_nmprad = self.calculate_roll_stiffness()

        # substituting sprung cg to front/rear distribution with rear/front sprung mass distribution
        roll_stiffness_sprung_f_nmprad = (
                roll_stiffness_f_nmprad - sprung_mass_distribution_f * sprung_mass_roll_moment_nmpg
        )
        roll_stiffness_sprung_r_nmprad = (
                roll_stiffness_r_nmprad - sprung_mass_distribution_r * sprung_mass_roll_moment_nmpg
        )
        return roll_stiffness_sprung_f_nmprad, roll_stiffness_sprung_r_nmprad

    def calculate_unsprung_load_transfer_sensitivity(self) -> tuple[float, float]:
        # load transfer sensitivity of unsprung mass
        unsprung_load_transfer_sensitivity_f_npg = (
                self.mass_unsprung_f_kg * 2 * 9.81 * self.cg_height_unsprung_f_m / self.track_f_m
        )
        unsprung_load_transfer_sensitivity_r_npg = (
                self.mass_unsprung_r_kg * 2 * 9.81 * self.cg_height_unsprung_r_m / self.track_r_m
        )
        return unsprung_load_transfer_sensitivity_f_npg, unsprung_load_transfer_sensitivity_r_npg

    def lateral_load_transfer_sensitivity_definitions(self) -> None:
        """
        Defines lateral load transfer sensitivities for the front and rear of the car in newtons
        seconds^2 / meter as are VehicleParam variable. This variable can be multiplied by
        lateral acceleration to calculate lateral load transfer to find the change in normal force
        on the tire at each axle

        All calculations are essentially from Pg 682 of Race Car Vehicle Dynamics

        Assumptions:
        - body roll angles are small (sin law)
        - roll centers and roll moment arms are static
        - the car is symmetric about its longitudinal plane
        """

        # define sprung mass distributions
        sprung_mass_distribution_f, sprung_mass_distribution_r = self.calculate_sprung_mass_distribution()

        roll_stiffness_sprung_f_nmprad, roll_stiffness_sprung_r_nmprad = self.calculate_roll_stiffness_sprung(
            sprung_mass_distribution_f=sprung_mass_distribution_f, sprung_mass_distribution_r=sprung_mass_distribution_r
        )

        (
            unsprung_load_transfer_sensitivity_f_npg,
            unsprung_load_transfer_sensitivity_r_npg,
        ) = self.calculate_unsprung_load_transfer_sensitivity()

        # define lateral load sensitivity in newtons of tire load change per g of lateral acceleration
        lat_load_transfer_sensitivity_f_npg = (
                self.weight_sprung_n
                / self.track_f_m
                * (
                        self.roll_gradient_radpg * roll_stiffness_sprung_f_nmprad / self.weight_sprung_n
                        + (sprung_mass_distribution_f * self.roll_center_height_f_m)
                )
                + unsprung_load_transfer_sensitivity_f_npg
        )
        lat_load_transfer_sensitivity_r_npg = (
                self.weight_sprung_n
                / self.track_r_m
                * (
                        self.roll_gradient_radpg * roll_stiffness_sprung_r_nmprad / self.weight_sprung_n
                        + (sprung_mass_distribution_r * self.roll_center_height_r_m)
                )
                + unsprung_load_transfer_sensitivity_r_npg
        )

        # converted to lateral load change in newtons per m/s^2 of lateral acceleration
        self.lat_load_transfer_sensitivity_f_ns2pm = lat_load_transfer_sensitivity_f_npg / 9.81
        self.lat_load_transfer_sensitivity_r_ns2pm = lat_load_transfer_sensitivity_r_npg / 9.81

    def tire_model_definitions(self) -> None:

        tire_model_info = [self.tire_file_and_version_f, self.tire_file_and_version_r]
        tire_model_definitions = [["front tire's", "tire_f"], ["rear tire's", "tire_r"]]

        for (filepath, tire_model_version), (tires, tire_model) in zip(tire_model_info, tire_model_definitions):
            if not filepath:
                qt_helper = pyqt_helper.Dialogs(__file__ + f"get {tires} TIR Model")
                filepath = str(qt_helper.select_file_dialog(accepted_file_types="*.TIR"))
            if tire_model_version == 52:
                setattr(self, tire_model, Mf52(filepath=filepath))
            elif tire_model_version == 61:
                setattr(self, tire_model, Mf61(filepath=filepath))
            else:
                raise AttributeError(
                    f"Define {tires} model version ({tire_model_version}) does not exist " f"in tires.py module"
                )

    def kinematic_definitions(self) -> None:
        if not self.kinematic_file:
            qt_helper = pyqt_helper.Dialogs(__file__ + 'Get Kinematics File')
            self.kinematic_file = str(qt_helper.select_file_dialog(accepted_file_types='*.xlsx'))

        kin_tables = KinematicData(self.kinematic_file)
        self.kinematics_l = KinematicEquations(kin_tables.left_side_kinematic_tables)
        self.kinematics_r = KinematicEquations(kin_tables.right_side_kinematic_tables)


if __name__ == "__main__":
    car = VehicleParameters()
    print(car)
