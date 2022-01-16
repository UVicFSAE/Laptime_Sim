"""
- For importing and dealing with kinematic sheets for the car
- Will likely be altered or supplemented with methods for GUI use
- For now, only works with specific formatting of current kinematic sheets
- All values as exported by lotus are for left side tire

Contributors: Nigel Swab
Created: 2021
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
from numpy import ndarray, rad2deg, deg2rad, reshape, meshgrid
from matplotlib import cm
from pandas import DataFrame as df, read_excel
from numba.typed import List

from lib.fit_equations import EquationFits, PolyCurve, PolySurface
from lib.print_colour import printy as print_colour
from lib.pyqt_helper import Dialogs

"""
Each SHEET NAME must match exactly with how the current program is written. 
Sheets [Name]: 
- [K_CAM_f] - Kinematic Camber front -> Index = Bump Travel [mm], Columns = Steering Rack Travel [mm], Data = Camber
- [K_CAM_r] - Kinematic Camber rear  -> Index = Bump Travel [mm]
- [K_TOE_f] - Kinematic Toe front -> Index = Bump Travel [mm], Columns = Steering Rack Travel [mm], Data = Toe
- [K_TOE_r] - Kinematic Toe rear -> Index = Bump Travel [mm]
- [R_CAM_f] - Roll Camber front -> Index = Roll Angle [deg]
- [R_CAM_r] - Roll Camber rear -> Index = Roll Angle [deg]
* Note that for each sheet, camber gets converted to inclination

Data is created by using defined motions in Lotus Suspension Analysis Shark. The current output format is 
for the left side wheel with standard SAE conventions:
- Toe-in = +, toe-out = -
- Camber

To integrate this cleanly into the rest of the program, this module will convert them to ISO coordinates 
for each side, opting to use inclination angle instead of Camber 

Refer to vehicle\Steering and Kinematics\KinSheetUV19.xlsx for general formatting

In the future, this module is likely to change significantly to accommodate GUI use
"""

# TODO: Module could use a fairly major refactor, kind of gross as it is
#  - Relies to heavily on a perfectly formatted excel workbook


@dataclass
class KinematicData:
    kinematic_tables: dict[str:df]
    left_side_kinematic_tables: dict[str:df]
    right_side_kinematic_tables: dict[str:df]

    def __init__(self, filepath: str):
        self._import_kinematic_data(filepath)
        self._format_kinematic_data()
        self._create_left_side_kinematic_table()
        self._create_right_side_kinematic_table()
        print("\nKinematic tables are ready! \n")

    def _import_kinematic_data(self, filepath) -> None:
        print("\nImporting kinematic Excel file... \n")
        print_colour("Ensure that kinematic tables are for the left side\n", colour="pale yellow")
        self.kinematic_tables = read_excel(filepath, sheet_name=None, header=0)

    def _format_kinematic_data(self) -> None:
        print("\t- Formatting kinematic_data...")

        # Convert all headings to lowercase and all 'CAM' to incln for consistent terminology
        self._format_kinematic_table_headings()

        for table in self.kinematic_tables.values():
            # Remove all NaN values
            table.dropna(how="all", axis="columns", inplace=True)
            # Convert all degree values to rad
            self._convert_kinematics_deg_to_rad(table)

        # Convert all bump values to m to retain SI units throughout repo
        self._convert_bump_mm_to_m()

    def _format_kinematic_table_headings(self):
        new_kinematic_tables = {}

        for table_name, table in self.kinematic_tables.items():

            if "CAM" in table_name:
                # Split 'CAM' from excel sheet/table title
                new_key_prefix, new_key_suffix = table_name.split("CAM", maxsplit=1)
                # Create a new table title (dictionary key) with incln instead
                new_table_key = f"{new_key_prefix}incln{new_key_suffix}".lower()
                # Add dataframe with new key to dictionary
                new_kinematic_tables[new_table_key] = table
            else:
                # Make all letters lowercase
                new_kinematic_tables[f"{table_name}".lower()] = table

        self.kinematic_tables = new_kinematic_tables

    def _convert_bump_mm_to_m(self) -> None:
        for table_name, table in self.kinematic_tables.items():
            if "k" in table_name:
                table.iloc[:, 0] /= 1000
                table.rename(
                    {"Bump Travel [mm]": "Bump Travel [m]"}, axis="columns", inplace=True
                )

    @staticmethod
    def _convert_kinematics_deg_to_rad(kinematic_table: df) -> None:
        for column_name in kinematic_table.columns:
            column_name_type = type(column_name)
            if column_name_type == str and "[deg]" in column_name:  # If column in degrees
                # Convert data from degrees to radians
                kinematic_table[column_name] = deg2rad(kinematic_table[column_name])
                # Change column name's unit
                parameter, unit = column_name.split(sep="[", maxsplit=1)
                new_column_name = f"{parameter}[rad]"
                kinematic_table.rename(columns={column_name: new_column_name}, inplace=True)
            elif column_name_type == str and "\\" in column_name:  # Else if: (ie. Bump Travel [mm] \ Rack Travel [mm])
                # The column title = the label left of '/'
                new_column_name, _ = column_name.split(sep="\\", maxsplit=1)
                kinematic_table.rename(columns={column_name: new_column_name}, inplace=True)
            elif column_name_type in [int, float]:  # If the column title is numeric
                # Convert data from degrees to radians
                kinematic_table[column_name] = deg2rad(kinematic_table[column_name])

    @staticmethod
    def _convert_kinematics_to_iso(kinematics: df) -> None:
        kinematics.iloc[:, 1:] *= -1

    def _create_left_side_kinematic_table(self):
        print("\t- Defining left_side_kinematic_tables...")
        self.left_side_kinematic_tables = {}
        for table_name, table in self.kinematic_tables.items():
            self.left_side_kinematic_tables[table_name] = table.copy(deep=True)
            # Convert to ISO definitions of inclination and toe
            self._convert_kinematics_to_iso(table)

    def _create_right_side_kinematic_table(self):
        # sourcery skip: remove-pass-elif
        print("\t- Defining right_side_kinematic_tables...")
        self.right_side_kinematic_tables = {}
        for table_name, table in self.kinematic_tables.items():
            self.right_side_kinematic_tables[table_name] = table.copy(deep=True)
            if table_name in ["k_incln_f", "k_toe_f"]:
                self.right_side_kinematic_tables[table_name].columns *= -1
            elif table_name in ["r_incln_f", "r_incln_r"]:
                self.right_side_kinematic_tables[table_name].iloc[:, 0] *= -1


class KinematicEquations:
    k_incln_f: EquationFits
    k_incln_r: EquationFits
    k_toe_f: EquationFits
    k_toe_r: EquationFits
    r_incln_f: EquationFits
    r_incln_r: EquationFits
    interp: bool

    def __init__(
        self,
        kinematic_table: dict[str:df],
        kin_incln_f_sfit: str = "poly33",
        kin_incln_r_cfit: str = "poly2",
        kin_toe_f_sfit: str = "poly14",
        kin_toe_r_cfit: str = "poly2",
        roll_incln_f_cfit: str = "poly2",
        roll_incln_r_cfit: str = "poly2",
        plot_fit=False,
    ):

        # Assign the fit/equation types for each kinematic table in the excel sheet
        kinematic_fit_types = {
            "k_incln_f": kin_incln_f_sfit,
            "k_incln_r": kin_incln_r_cfit,
            "k_toe_f": kin_toe_f_sfit,
            "k_toe_r": kin_toe_r_cfit,
            "r_incln_f": roll_incln_f_cfit,
            "r_incln_r": roll_incln_r_cfit,
            }

        for kinematic, fit in kinematic_fit_types.items():
            parameter = "Inclination Angle" if "incln" in kinematic else "Toe"
            if fit in dir(PolyCurve):
                setattr(self, kinematic, PolyCurve(fit, parameter))
            elif fit in dir(PolySurface):
                setattr(self, kinematic, PolySurface(fit, parameter))

        self.fit_kinematics(kinematic_table, plot_fit)

    def fit_kinematics(self, kinematic_tables: dict[str:df], plot=False) -> None:

        for kinematic in vars(self):
            kinematic_obj = getattr(self, kinematic)
            if isinstance(kinematic_obj, PolyCurve):
                x, x_param, y, y_param = self._prepare_kin_curve_data(kinematic_tables[kinematic])
                kinematic_obj.fit(x, y)
                if plot:
                    y_pred = kinematic_obj.calculate(x)
                    self.plot_kin_curve_fit(x, y, y_pred, x_param, y_param)
            elif isinstance(kinematic_obj, PolySurface):
                X, Y, x, y, z = self._prepare_kin_surface_data(kinematic_tables[kinematic])
                kinematic_obj.fit([x, y], z)
                if plot:
                    z_pred = kinematic_obj.calculate([x, y])
                    self.plot_kin_surface_fit(X, Y, z, z_pred, param=kinematic_obj.fit_parameter)

    @staticmethod
    def _prepare_kin_curve_data(data: df) -> tuple[ndarray, str, ndarray, str]:

        # Extract the first column of data and it's title (Rack Travel [mm], Bump Travel [mm], Roll Angle [rad])
        x, x_param = data.iloc[:, 0], data.columns[0]
        # Extract the second column of data and it's title (Wheel Angle [rad], Camber Angle [rad])
        y, y_param = data.iloc[:, 1], data.columns[1]

        return x.astype("float64"), x_param, y.astype("float64"), y_param

    @staticmethod
    def plot_kin_curve_fit(x: ndarray, y: ndarray, y_pred: ndarray, x_param: str = "", y_param: str = "") -> None:

        # Convert to degrees for readability
        x = rad2deg(x) if "rad" in x_param else x
        y = rad2deg(y) if "rad" in y_param else y
        y_pred = rad2deg(y_pred) if "rad" in y_param else y

        # Create plots
        fig, ax = plt.subplots()
        ax.plot(x, y, marker=".")
        ax.plot(x, y_pred, c="r", linestyle="-")
        # TODO: Add labels and titles
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        plt.show()

    @staticmethod
    def _prepare_kin_surface_data(data: df) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:

        # Extract the first row of data (Steering Rack Travel [m]), skipping the column & row label
        steering_rack_data_mm = data.columns[1:].to_numpy()
        # Extract first column of data (Bump Travel [m])
        bump_travel_data_m = data.iloc[:, 0].to_numpy()
        # Extract camber/toe data
        z = data.iloc[:, 1:].to_numpy()

        # Create a 2D mesh
        Y, X = meshgrid(steering_rack_data_mm, bump_travel_data_m)  # Y = Rack Travel [mm], X = Bump [mm]
        # Flatten 2D mesh for equation fitting
        x = X.flatten().astype("float64")
        y = Y.flatten().astype("float64")
        z = z.flatten().astype("float64")

        return X.astype("float64"), Y.astype("float64"), x, y, z

    @staticmethod
    def plot_kin_surface_fit(X: ndarray, Y: ndarray, z: ndarray, z_pred: ndarray, param: str = "") -> None:

        # Reshape z for correct surface plot format
        Z_pred = reshape(z_pred, X.shape)

        # Convert to degrees for readability
        Z_pred = rad2deg(Z_pred)
        z = rad2deg(z)

        # Create plots
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z_pred, cmap=cm.get_cmap("Spectral"), linewidth=0, antialiased=False)
        ax.scatter3D(X, Y, z)

        # Titles and labels
        plt.title(f"{param} Map")
        ax.set_xlabel("Bump Travel [m]")
        ax.set_ylabel("Steering Rack Travel [mm]")
        ax.set_zlabel(f"{param} [deg]")
        plt.show()

    def calculate_wheel_angles(
        self,
        bump_travel_f_m: float,
        bump_travel_r_m: float,
        rack_travel_mm: float,
        roll_angle_rad: float,
        static_incln_f_rad: float,
        static_incln_r_rad: float,
        static_toe_f_rad: float,
        static_toe_r_rad: float,
    ) -> tuple[float, float, float, float]:
        incln_f_rad, incln_r_rad = self.calculate_inclination_angles(
            bump_travel_f_m=bump_travel_f_m,
            bump_travel_r_m=bump_travel_r_m,
            rack_travel_mm=rack_travel_mm,
            roll_angle_rad=roll_angle_rad,
            static_incln_f_rad=static_incln_f_rad,
            static_incln_r_rad=static_incln_r_rad,
        )
        toe_f_rad, toe_r_rad = self.calculate_toe_angles(
            bump_travel_f_m=bump_travel_f_m,
            bump_travel_r_m=bump_travel_r_m,
            rack_travel_mm=rack_travel_mm,
            static_toe_f_rad=static_toe_f_rad,
            static_toe_r_rad=static_toe_r_rad,
        )
        return incln_f_rad, incln_r_rad, toe_f_rad, toe_r_rad

    def calculate_inclination_angles(
        self,
        bump_travel_f_m: float,
        bump_travel_r_m: float,
        rack_travel_mm: float,
        roll_angle_rad: float,
        static_incln_f_rad: float,
        static_incln_r_rad: float,
    ) -> tuple[float, float]:

        # Calculate front inclination angle
        front_kin_incln_rad = self.k_incln_f.calculate([bump_travel_f_m, rack_travel_mm])
        front_roll_incln_rad = self.r_incln_f.calculate(roll_angle_rad)
        front_incln_rad = front_kin_incln_rad + front_roll_incln_rad + static_incln_f_rad
        # Calculate rear inclination angle
        rear_kin_incln_rad = self.k_incln_r.calculate(bump_travel_r_m)
        rear_roll_incln_rad = self.r_incln_r.calculate(roll_angle_rad)
        rear_incln_rad = rear_kin_incln_rad + rear_roll_incln_rad + static_incln_r_rad

        return front_incln_rad, rear_incln_rad

    def calculate_toe_angles(
        self,
        bump_travel_f_m: float,
        bump_travel_r_m: float,
        rack_travel_mm: float,
        static_toe_f_rad: float,
        static_toe_r_rad: float,
    ) -> tuple[float, float]:

        # Calculate front toe angle
        front_toe_rad = self.k_toe_f.calculate([bump_travel_f_m, rack_travel_mm]) + static_toe_f_rad
        # Calculate rear toe angle
        rear_toe_rad = self.k_toe_r.calculate(bump_travel_r_m) + static_toe_r_rad

        return front_toe_rad, rear_toe_rad


if __name__ == "__main__":

    # For debugging. Set filepath = '' if you want to use file explorer to choose kinematic file
    file = "vehicle\steering_and_kinematics\KinSheetUV19.xlsx"
    if not file:
        # Choose and load tire model
        qt_helper = Dialogs(__file__ + "Get Kinematics File")
        file = str(qt_helper.select_file_dialog(accepted_file_types="*.xlsx"))

    kin_tables = KinematicData(file)
    kinematics_left = KinematicEquations(kin_tables.left_side_kinematic_tables, plot_fit=False)
    kinematics_right = KinematicEquations(kin_tables.right_side_kinematic_tables, plot_fit=True)
