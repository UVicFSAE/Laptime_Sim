'''
- For importing and dealing with kinematic sheets for the car
- Will likely be replaced or suplemented with methods for GUI use
- For now, only works with specific formatting of current kinematic sheets
- All values as exported by lotus are for left side

Contributors: Nigel Swab
Created: 2021
'''

import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from pandas import read_excel
from scipy.optimize import curve_fit
from matplotlib import cm

import fit_equations
from lib.pyqt_helper import Dialogs

'''
Sheets: 
- Steering - Ackerman Geometry 
- KCAMf - Kinematic Camber -> Index = Bump Travel [mm], Columns = Steering Rack Travel [mm]
- KCAMr
- KTOEf
- KTOEr
- RCAMf
- RCAMr
'''


def import_kinematics(file: str) -> dict:
    print(f'Importing kinematic Excel file...\n')
    kinematics = read_excel(file, sheet_name=None, header=0)

    print('Converting kinematics from degrees to radian...\n')
    for sheet in kinematics:
        # Remove all NaN values
        kinematics[sheet].dropna(how='all', axis='columns', inplace=True)
        # Convert all degree values to rad
        kinematics[sheet] = convert_kinematics_deg_to_rad(kinematics, sheet)
    print('Kinematics are ready for use.\n')
    return kinematics


def convert_kinematics_deg_to_rad(kinematics: df, sheet: str) -> df:

    for column_name in kinematics[sheet].columns:
        column_name_type = type(column_name)
        if column_name_type == str and '[deg]' in column_name:  # If column in degrees
            # Convert data from degrees to radians
            kinematics[sheet][column_name] = np.deg2rad(kinematics[sheet][column_name])
            # Change column name's unit
            parameter, unit = column_name.split(sep='[', maxsplit=1)
            new_column_name = f'{parameter}[rad]'
            kinematics[sheet].rename(columns={column_name: new_column_name}, inplace=True)
        elif column_name_type == str and "\\" in column_name:  # Else if ... (ie. Bump Travel [mm] \ Rack Travel [mm])
            # The column title = the label left of '/'
            new_column_name, _ = column_name.split(sep='\\', maxsplit=1)
            kinematics[sheet].rename(columns={column_name: new_column_name}, inplace=True)
        elif column_name_type == int or column_name_type == float:  # If the column title is numeric
            # Convert data from degrees to radians
            kinematics[sheet][column_name] = np.deg2rad(kinematics[sheet][column_name])

    return kinematics[sheet]


def print_kinematics_tables(kinematics: dict[df]) -> None:
    for sheet in kinematics:
        print(f'{kinematic_data[sheet]}\n')


def fit_kin_surface(data: df, equation, fit_parameter: str = '', plot=False) -> list:

    X, Y, z = prepare_kin_surface_data(data)

    # Fit kinematic surface with the defined fitting equation
    coefficients, covariance = curve_fit(equation, [X.flatten(), Y.flatten()], z)

    if plot:
        z_pred = equation([X.flatten(), Y.flatten()], *coefficients)
        _plot_kin_surface_fit(X, Y, z, z_pred, param=fit_parameter)

    return coefficients


def fit_kin_curve(data:df, fit_type, fit_parameter: str = '', plot=False) -> list:
    pass


def prepare_kin_surface_data(data: df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Extract the first row of data (Steering Rack Travel [mm]), skipping the column & row label
    steering_rack_data_mm = data.columns[1:].to_numpy()
    # Extract first column of data (Bump Travel [mm])
    bump_travel_data_mm = data['Bump Travel [mm]'].to_numpy()
    # Extract camber/toe data
    z = data.iloc[:, 1:].to_numpy()

    # Create a 2D mesh and flatten
    steering_rack_mm, bump_mm = np.meshgrid(steering_rack_data_mm, bump_travel_data_mm)
    z = z.flatten()

    return bump_mm.astype('float64'), steering_rack_mm.astype('float64'), z.astype('float64')


def prepare_kin_curve_data(data:df) -> np.ndarray:
    pass


def _plot_kin_surface_fit(X: np.ndarray, Y: np.ndarray, z: np.ndarray, z_pred: np.ndarray, param: str = '') -> None:

    # Reshape z for correct surface plot format
    Z_pred = np.reshape(z_pred, X.shape)

    # Convert to degrees for readability
    Z_pred = np.rad2deg(Z_pred)
    z = np.rad2deg(z)

    # Create plots
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z_pred, cmap=cm.get_cmap('Spectral'), linewidth=0, antialiased=False)
    ax.scatter3D(X, Y, z)

    # Titles and labels
    plt.title(f'{param} Map')
    ax.set_xlabel('Bump Travel [mm]')
    ax.set_label('Steering Rack Travel [mm]')
    ax.set_zlabel(f'{param} [deg]')
    plt.show()


def _plot_kin_curve_fit(x: np.ndarray,  y: np.ndarray) -> None:
    pass


if __name__ == "__main__":

    # For debugging. Set filepath = '' if you want to use file explorer to choose kinematic file
    filepath = 'vehicle\Steering and Kinematics\KinSheetUV19.xlsx'
    if not filepath:
        # Choose and load tire model
        qt_helper = Dialogs(__file__ + 'Get Kinematics File')
        filepath = str(qt_helper.select_file_dialog(accepted_file_types='*.xlsx'))
    kinematic_data = import_kinematics(filepath)

    # Print for debugging
    # print_kinematics_tables(kinematic_data)
    KCAMf_coeffs = fit_kin_surface(kinematic_data['KCAMf'], equation=poly33, fit_parameter='Camber', plot=True)
    KTOEf_coeffs = fit_kin_surface(kinematic_data['KTOEf'], equation=poly12, fit_parameter='Toe', plot=True)
