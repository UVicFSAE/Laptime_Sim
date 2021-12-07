'''
- For importing and dealing with kinematic sheets for the car
- Will likely be altered or supplemented with methods for GUI use
- For now, only works with specific formatting of current kinematic sheets
- All values as exported by lotus are for left side tire

Contributors: Nigel Swab
Created: 2021
'''
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pandas import DataFrame as df, read_excel

from lib.fit_equations import EquationFits, PolyCurve, PolySurface
from lib.print_colour import printy as print_colour
from lib.pyqt_helper import Dialogs
'''
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
'''

# TODO: Module could use a fairly major refactor, kind of gross as it is


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
        print('\nKinematic tables are ready! \n')

    def _import_kinematic_data(self, filepath) -> None:
        print('\nImporting kinematic Excel file... \n')
        print_colour('Ensure that kinematic tables are for the left side\n',
                     colour='pale yellow')
        self.kinematic_tables = read_excel(filepath, sheet_name=None, header=0)

    def _format_kinematic_data(self) -> None:
        print('\t- Formatting kinematic_data...')

        # Convert all headings to lowercase and all 'CAM' to incln to retain terminology consistency
        self._format_kinematic_table_headings()

        for table in self.kinematic_tables:
            # Remove all NaN values
            self.kinematic_tables[table].dropna(how='all',
                                                axis='columns',
                                                inplace=True)
            # Convert all degree values to rad
            self._convert_kinematics_deg_to_rad(self.kinematic_tables[table])

    def _format_kinematic_table_headings(self):
        kinematic_table_keys = list(self.kinematic_tables.keys())
        for table_key in kinematic_table_keys:
            if 'CAM' in table_key:
                # Split 'CAM' from excel sheet/table title
                new_key_prefix, new_key_suffix = table_key.split('CAM',
                                                                 maxsplit=1)
                # Create a new table title (dictionary key) with incln instead
                new_table_key = f'{new_key_prefix}incln{new_key_suffix}'.lower()
                # Add dataframe with new key to dictionary
                self.kinematic_tables[new_table_key] = self.kinematic_tables[
                    table_key]
            else:
                # Make all letters lowercase
                self.kinematic_tables[
                    f'{table_key}'.lower()] = self.kinematic_tables[table_key]
            # Delete dataframe with the old key
            del self.kinematic_tables[table_key]

    @staticmethod
    def _convert_kinematics_deg_to_rad(kinematic_table: df) -> None:
        for column_name in kinematic_table.columns:
            column_name_type = type(column_name)
            if column_name_type == str and '[deg]' in column_name:  # If column in degrees
                # Convert data from degrees to radians
                kinematic_table[column_name] = np.deg2rad(
                    kinematic_table[column_name])
                # Change column name's unit
                parameter, unit = column_name.split(sep='[', maxsplit=1)
                new_column_name = f'{parameter}[rad]'
                kinematic_table.rename(columns={column_name: new_column_name},
                                       inplace=True)
            elif column_name_type == str and "\\" in column_name:  # Else if: (ie. Bump Travel [mm] \ Rack Travel [mm])
                # The column title = the label left of '/'
                new_column_name, _ = column_name.split(sep='\\', maxsplit=1)
                kinematic_table.rename(columns={column_name: new_column_name},
                                       inplace=True)
            elif column_name_type in [int,
                                      float]:  # If the column title is numeric
                # Convert data from degrees to radians
                kinematic_table[column_name] = np.deg2rad(
                    kinematic_table[column_name])

    @staticmethod
    def _convert_kinematics_to_iso(kinematics: df) -> None:
        kinematics.iloc[:, 1:] *= -1

    def _create_left_side_kinematic_table(self):
        print('\t- Defining left_side_kinematic_tables...')
        self.left_side_kinematic_tables = {}
        for table in self.kinematic_tables:
            self.left_side_kinematic_tables[table] = self.kinematic_tables[
                table].copy(deep=True)
            # Convert to ISO definitions of inclination and toe
            self._convert_kinematics_to_iso(
                self.left_side_kinematic_tables[table])

    def _create_right_side_kinematic_table(self):
        # sourcery skip: remove-pass-elif
        print('\t- Defining right_side_kinematic_tables...')
        self.right_side_kinematic_tables = {}
        for table in self.kinematic_tables:
            self.right_side_kinematic_tables[table] = self.kinematic_tables[
                table].copy(deep=True)
            if table in ['k_incln_f', 'k_toe_f']:
                self.right_side_kinematic_tables[table].columns *= -1
            elif table in ['r_incln_f', 'r_incln_r']:
                self.right_side_kinematic_tables[table].iloc[:, 0] *= -1
            elif table in ['k_incln_r', 'k_toe_r']:
                pass


class KinematicEquations:
    k_incln_f: EquationFits
    k_incln_r: EquationFits
    k_toe_f: EquationFits
    k_toe_r: EquationFits
    r_incln_f: EquationFits
    r_incln_r: EquationFits

    def __init__(self,
                 kinematic_table: dict[str:df],
                 kin_incln_f_sfit: str = 'poly33',
                 kin_incln_r_cfit: str = 'poly2',
                 kin_toe_f_sfit: str = 'poly12',
                 kin_toe_r_cfit: str = 'poly2',
                 roll_incln_f_cfit: str = 'poly2',
                 roll_incln_r_cfit: str = 'poly2',
                 plot_fit=False):

        # Assign the fit/equation types for each kinematic table in the excel sheet
        kinematic_fit_types = {
            'k_incln_f': kin_incln_f_sfit,
            'k_incln_r': kin_incln_r_cfit,
            'k_toe_f': kin_toe_f_sfit,
            'k_toe_r': kin_toe_r_cfit,
            'r_incln_f': roll_incln_f_cfit,
            'r_incln_r': roll_incln_r_cfit
        }

        for kinematic, fit in kinematic_fit_types.items():
            parameter = 'Inclination Angle' if 'incln' in kinematic else 'Toe'
            if fit in dir(PolyCurve):
                setattr(self, kinematic, PolyCurve(fit, parameter))
            elif fit in dir(PolySurface):
                setattr(self, kinematic, PolySurface(fit, parameter))

        self.fit_kinematics(kinematic_table, plot_fit)

    def calculate_inclination_angles(self, bump_travel_mm, rack_travel_mm,
                                     roll_angle_rad):

        # Calculate front inclination angle
        front_kin_incln_rad = self.k_incln_f.calculate(
            [bump_travel_mm, rack_travel_mm])
        front_roll_incln_rad = self.r_incln_f.calculate(roll_angle_rad)
        front_incln_rad = front_kin_incln_rad + front_roll_incln_rad
        # Calculate rear inclination angle
        rear_kin_incln_rad = self.k_incln_r.calculate(
            [bump_travel_mm, rack_travel_mm])
        rear_roll_incln_rad = self.r_incln_r.calculate(roll_angle_rad)
        rear_incln_rad = rear_kin_incln_rad + rear_roll_incln_rad

        return front_incln_rad, rear_incln_rad

    def calculate_wheel_angles(self, bump_travel_mm, rack_travel_mm):

        # Calculate front toe angle
        front_toe_rad = self.k_toe_f.calculate([bump_travel_mm, rack_travel_mm])
        # Calculate rear toe angle
        rear_toe_rad = self.k_toe_r.calculate([bump_travel_mm, rack_travel_mm])

        return front_toe_rad, rear_toe_rad

    def fit_kinematics(self,
                       kinematic_tables: dict[str:df],
                       plot=False) -> None:

        for kinematic in vars(self):
            kinematic_obj = getattr(self, kinematic)
            if type(kinematic_obj) == PolySurface:
                X, Y, x, y, z = self.prepare_kin_surface_data(
                    kinematic_tables[kinematic])
                kinematic_obj.fit([x, y], z)
                if plot:
                    z_pred = kinematic_obj.calculate([x, y])
                    self.plot_kin_surface_fit(X,
                                              Y,
                                              z,
                                              z_pred,
                                              param=kinematic_obj.fit_parameter)
            elif type(kinematic_obj) == PolyCurve:
                x, x_param, y, y_param = self.prepare_kin_curve_data(
                    kinematic_tables[kinematic])
                kinematic_obj.fit(x, y)
                if plot:
                    y_pred = kinematic_obj.calculate(x)
                    self.plot_kin_curve_fit(x, y, y_pred, x_param, y_param)

    @staticmethod
    def prepare_kin_curve_data(
            data: df) -> tuple[np.ndarray, str, np.ndarray, str]:

        # Extract the first column of data and it's title (Rack Travel [mm], Bump Travel [mm], Roll Angle [rad])
        x, x_param = data.iloc[:, 0], data.columns[0]
        # Extract the second column of data and it's title (Wheel Angle [rad], Camber Angle [rad])
        y, y_param = data.iloc[:, 1], data.columns[1]

        return x.astype('float64'), x_param, y.astype('float64'), y_param

    @staticmethod
    def prepare_kin_surface_data(
        data: df
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Extract the first row of data (Steering Rack Travel [mm]), skipping the column & row label
        steering_rack_data_mm = data.columns[1:].to_numpy()
        # Extract first column of data (Bump Travel [mm])
        bump_travel_data_mm = data.iloc[:, 0].to_numpy()
        # Extract camber/toe data
        z = data.iloc[:, 1:].to_numpy()

        # Create a 2D mesh
        Y, X = np.meshgrid(
            steering_rack_data_mm,
            bump_travel_data_mm)  # Y = Rack Travel [mm], X = Bump [mm]
        # Flatten 2D mesh for equation fitting
        x = X.flatten().astype('float64')
        y = Y.flatten().astype('float64')
        z = z.flatten().astype('float64')

        return X.astype('float64'), Y.astype('float64'), x, y, z

    @staticmethod
    def plot_kin_surface_fit(X: np.ndarray,
                             Y: np.ndarray,
                             z: np.ndarray,
                             z_pred: np.ndarray,
                             param: str = '') -> None:

        # Reshape z for correct surface plot format
        Z_pred = np.reshape(z_pred, X.shape)

        # Convert to degrees for readability
        Z_pred = np.rad2deg(Z_pred)
        z = np.rad2deg(z)

        # Create plots
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X,
                               Y,
                               Z_pred,
                               cmap=cm.get_cmap('Spectral'),
                               linewidth=0,
                               antialiased=False)
        ax.scatter3D(X, Y, z)

        # Titles and labels
        plt.title(f'{param} Map')
        ax.set_xlabel('Bump Travel [mm]')
        ax.set_ylabel('Steering Rack Travel [mm]')
        ax.set_zlabel(f'{param} [deg]')
        plt.show()

    @staticmethod
    def plot_kin_curve_fit(x: np.ndarray,
                           y: np.ndarray,
                           y_pred: np.ndarray,
                           x_param: str = '',
                           y_param: str = '') -> None:

        # Convert to degrees for readability
        x = np.rad2deg(x) if 'rad' in x_param else x
        y = np.rad2deg(y) if 'rad' in y_param else y
        y_pred = np.rad2deg(y_pred) if 'rad' in y_param else y

        # Create plots
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='.')
        ax.plot(x, y_pred, c='r', linestyle='-')
        # TODO: Add labels and titles
        plt.show()


if __name__ == "__main__":

    # For debugging. Set filepath = '' if you want to use file explorer to choose kinematic file
    file = 'vehicle\Steering and Kinematics\KinSheetUV19.xlsx'
    if not file:
        # Choose and load tire model
        qt_helper = Dialogs(__file__ + 'Get Kinematics File')
        file = str(qt_helper.select_file_dialog(accepted_file_types='*.xlsx'))

    kin_tables = KinematicData(file)
    kinematics_left = KinematicEquations(kin_tables.left_side_kinematic_tables,
                                         plot_fit=True)
    kinematics_right = KinematicEquations(
        kin_tables.right_side_kinematic_tables, plot_fit=True)
