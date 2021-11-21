'''
- For importing and dealing with kinematic sheets for the car
- Will likely be replaced or suplemented with methods for GUI use
- For now, only works with specific formatting of current kinematic sheets
- All values as exported by lotus are for left side

Created: Nigel Swab, 2021
'''

import numpy as np
from pandas import DataFrame as df
from pandas import read_excel

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


def import_kinematics(file: str) -> dict[df]:
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


def view_kinematics(kinematics: dict[df]) -> None:
    for sheet in kinematics:
        print(f'{kinematic_data[sheet]}\n')


if __name__ == "__main__":

    # Choose and load tire model
    qt_helper = Dialogs(__file__ + 'Get Kinematics File')
    filepath = str(qt_helper.select_file_dialog(accepted_file_types='*.xlsx'))
    kinematic_data = import_kinematics(filepath)

    # Print for debugging
    view_kinematics(kinematic_data)
