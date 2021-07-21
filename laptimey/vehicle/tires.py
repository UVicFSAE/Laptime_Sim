"""
This file contains the tire model classes and their functions for the calculation of tire forces

- alpha = slip angle [deg]
- kappa = slip ratio [-]
- gamma = camber [deg]

Created 2021, Contributors: Nigel Swab
"""
import numpy as np
from dataclasses import dataclass

from lib.pyqt_helper import Dialogs


class PacejkaMf61:
    
    def __init__(self, filename):

        # # List of parameters to pull from .TIR files
        # self.list_of_model_info = []
        # self.list_of_constraints = []
        # self.list_of_coefficients = []
        #
        # # Dictionaries to store tire model data
        # self.model_info = dict()
        # self.constraints = dict()
        # self.coefficient_descriptions = dict()
        #
        # # Load tire model data
        # self._load_tire_model_metadata(self)
        # self._load_tire_model(self, filename)

        # Initializations
        self.coeffs.Fz0 = self.coeffs.FNOMIN * self.coeffs.LFZ0

    def f_y(self, alpha, kappa, gamma, Fz, tire_pressure):
        """
        Calculates and returns the lateral force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        camber (gamma), normal force  (Fz), and tire pressure
        """

        # Initializations
        dfz = (Fz - self.coeffs.Fz0) / self.coeffs.Fz0
        dpi = (tire_pressure - self.coeffs.NOMPRES) / self.coeffs.NOMPRES

        # Pure slip
        SVyg = Fz * (self.coeffs.PVY3 + self.coeffs.PVY4 * dfz) * gamma * self.coeffs.LKY * self.coeffs.LMUY
        SVy0 = Fz * (self.coeffs.PVY1 + self.coeffs.PVY2 * dfz) * self.coeffs.LVY * self.coeffs.LMUY
        SVy = SVy0 + SVyg
        Kyg = (self.coeffs.PKY6 + self.coeffs.PKY7 * dfz) * (1 + self.coeffs.PPY5 * dpi) * Fz * self.coeffs.LKYC
        Kya = (1 - self.coeffs.PKY3 * abs(gamma)) * self.coeffs.PKY1 * self.coeffs.Fz0 * (1 + self.coeffs.PPY1 * dpi) * \
               np.sin(self.coeffs.PKY4 * np.arctan(Fz / ((self.coeffs.PKY2 + self.coeffs.PKY5 * np.power(gamma, 2))
                                                  * (1 + self.coeffs.PPY2 * dpi) * self.coeffs.Fz0))) * self.coeffs.LKY
        SHy0 = (self.coeffs.PHY1 + self.coeffs.PHY2 * dfz) * self.coeffs.LHY
        SHyg = (Kyg * gamma - SVyg) / Kya
        SHy = SHy0 + SHyg
        alpha_y = alpha + SHy
        Ey = (self.coeffs.PEY1 + self.coeffs.PEY2 * dfz) * \
             (1 + self.coeffs.PEY5 * np.power(gamma, 2) - (self.coeffs.PEY3 + self.coeffs.PEY4 * gamma) * np.sign(alpha_y)) * self.coeffs.LEY
        Mewy = (self.coeffs.PDY1 + self.coeffs.PDY2 * dfz) * (1 - self.coeffs.PDY3 * np.power(gamma, 2)) * \
               (1 + self.coeffs.PPY3 * dpi + self.coeffs.PPY4 * np.power(gamma, 2)) * self.coeffs.LMUY
        Dy = Mewy * Fz
        Cy = self.coeffs.PCY1 * self.coeffs.LCY
        By = Kya / (Cy * Dy)
        SVyk = 0
        Gyk = 1

        # Combined Slip (if there is a slip ratio
        if kappa:
            SHyk = self.coeffs.RHY1 + self.coeffs.RHY2 * dfz
            Eyk = self.coeffs.REY1 + self.coeffs.REY2 * dfz
            Cyk = self.coeffs.RCY1
            Byk = (self.coeffs.RBY1 + self.coeffs.RBY4 * np.power(gamma, 2)) * \
                  np.cos( np.arctan(self.coeffs.RBY2 * (alpha - self.coeffs.RBY3))) * self.coeffs.LYKA
            kappa_s = kappa + SHyk
            Gyk = (np.cos(Cyk * np.arctan(Byk * kappa_s - Eyk * (Byk * kappa_s - np.arctan(Byk * kappa_s))))) / \
                  (np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk)))))
            DVyk = Mewy * Fz * (self.coeffs.RVY1 + self.coeffs.RVY2 * dfz + self.coeffs.RVY3 * gamma) * \
                   np.cos(np.arctan(self.coeffs.RVY4 * alpha))
            SVyk = DVyk * np.sin(self.coeffs.RVY5 * np.arctan(self.coeffs.RVY6)) * self.coeffs.LVYK

        # Pure Lateral Force
        Fyp = Dy * np.sin(Cy * np.arctan(By * alpha_y - Ey * (By * alpha_y - np.arctan(By * alpha_y)))) + SVy

        # Combined Lateral Force ( Fyp == Fy if kappa == 0)
        Fy = Gyk * Fyp + SVyk
        return Fy

    def f_x(self, alpha, kappa, gamma, Fz, tire_pressure):
        """
        Calculates and returns the longitudinal force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        camber (gamma), normal force  (Fz), and tire pressure
        """

        # Initializations
        dfz = (Fz - self.coeffs.Fz0) / self.coeffs.Fz0
        dpi = (tire_pressure - self.coeffs.NOMPRES) / self.coeffs.NOMPRES

        # Pure Slip
        SVx = (self.coeffs.PVX1 + self.coeffs.PVX2 * dfz) * Fz * self.coeffs.LVX * self.coeffs.LMUX
        SHx = (self.coeffs.PHX1 + self.coeffs.PHX2 * dfz) * self.coeffs.LHX
        kappa_x = kappa + SHx
        Kxk = (self.coeffs.PKX1 + self.coeffs.PKX2 * dfz) * np.exp(self.coeffs.PKX3 * dfz) * \
              (1 + self.coeffs.PPX1 * dpi + self.coeffs.PPX2 * np.power(gamma, 2)) * Fz * self.coeffs.LKX
        Ex = (self.coeffs.PEX1 + self.coeffs.PEX2 * dfz + self.coeffs.PEX3 * np.power(gamma, 2)) * \
             (1 - self.coeffs.PEX4 * np.sign(kappa_x)) * self.coeffs.LEX
        Mewx = (self.coeffs.PDX1 + self.coeffs.PDX2 * dfz) * (1 - self.coeffs.PDX3 * np.power(gamma, 2)) * \
               (1 + self.coeffs.PPX3 * dpi + self.coeffs.PPX4 * np.power(gamma, 2)) * Fz * self.coeffs.LMUX
        Dx = Mewx * Fz
        Cx = self.coeffs.PCX1 * self.coeffs.LCX
        Bx = Kxk / (Cx * Dx)
        Gxa = 1

        # Combined Slip
        if alpha:
            SHxa = self.coeffs.RHX1
            Exa = self.coeffs.REX1 + self.coeffs.REX2 * dfz
            Cxa = self.coeffs.RCX1
            Bxa = (self.coeffs.RBX1 + self.coeffs.RBX3 * np.power(gamma, 2)) * np.cos(np.arctan(self.coeffs.RBX2 * kappa)) * self.coeffs.LXAL
            alpha_s = alpha + SHxa
            Gxa = (np.cos(Cxa * np.arctan(Bxa * alpha_s - Exa * (Bxa * alpha_s - np.arctan(Bxa * alpha_s))))) / \
                  (np.cos(Cxa * np.arctan(Bxa * SHxa - Exa * (Bxa * SHxa - np.arctan(Bxa * SHxa)))))

        # Longitudinal Force (N)
        Fx = (Dx * np.sin(Cx * np.arctan(Bx * kappa_x - Ex * (Bx * kappa_x - np.arctan(Bx * kappa_x)))) + SVx) * Gxa
        return Fx

    def m_z(self, alpha_M, kappa, gamma, Fz, tire_pressure):
        """
        Calculates and returns the aligning moment produced by a tire given a slip angle (alpha), slip ratio (kappa),
        camber (gamma), normal force  (Fz), and tire pressure
        """

        # Initializations
        dfz = (Fz - self.coeffs.Fz0) / self.coeffs.Fz0
        dpi = (tire_pressure - self.coeffs.NOMPRES) / self.coeffs.NOMPRES

        # Residual Moment (Mzr)
        Dr = ((self.coeffs.QDZ6 + self.coeffs.QDZ7 * dfz) * self.coeffs.LRES + (self.coeffs.QDZ8 + self.coeffs.QDZ9 * dfz) *
              (1 - self.coeffs.PPZ2 * dpi) * gamma * self.coeffs.LKZ + (self.coeffs.QDZ10 + self.coeffs.QDZ11 * dfz) *
              gamma * np.abs(gamma) * self.coeffs.LKZ) * Fz * self.coeffs.UNLOADED_RADIUS * self.coeffs.LMUY
        Br = self.coeffs.QBZ9 * self.coeffs.LKY / self.coeffs.LMUY + self.coeffs.QBZ10 * By * Cy
        Mzr = Dr * np.cos(np.arctan(Br * alpha_req)) * np.cos(alpha_M)

    
    def m_x(self):
        pass

    def plot_force(self):
        pass

    @staticmethod
    def _load_tire_model(self, filepath: str) -> None:
        """ Takes the passed in file path, extracts the necessary data, and populates the dictionary with both the key
        and data values.
        """

        print('Loading tire model... \n')

        # Load all lines of data
        with open(filepath, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Ensure it's not a header or divider line
            if line[0] not in ['[', '$', '!', '\n', '\t'] and line[1] not in ['[', '$', '!', '\n', '\t']:
                # Remove whitespaces
                line = "".join(line.split())
                # Separate the parameter key and it's value
                key, value = line.split(sep='=', maxsplit=1)

                # Evaluate if param is model info, constraint, or coefficient -> remove description of param and save
                if key in self.list_of_model_info:
                    value, _ = value.split(sep='$', maxsplit=1)
                    self.model_info[key] = value
                elif key in self.list_of_constraints:
                    value, _ = value.split(sep='$', maxsplit=1)
                    setattr(self, key, float(value))
                elif key in self.list_of_coefficients:
                    value, description = value.split(sep='$', maxsplit=1)
                    setattr(self, key, float(value))
                    self.coefficient_descriptions[key] = description

    @staticmethod
    def _load_tire_model_metadata(self):
        """
        Lists of data parameters we care to pull from .TIR files
        """
        self.list_of_model_info = ['FITTYP', 'WIDTH', 'ASPECT_RATIO', 'VERTICAL_STIFFNESS']

        self.list_of_constraints = ['PRESMIN', 'PRESMAX', 'KPUMIN', 'KPUMAX', 'APLMIN', 'ALPMAX', 'CAMMIN', 'CAMMAX']

        self.list_of_coefficients = ['FNOMIN', 'UNLOADED_RADIUS', 'LONGVL', 'NOMPRES', 'LFZ0', 'LCX', 'LMUX',
                                     'LEX', 'LKX', 'LHX', 'LVX', 'LCY', 'LMUY', 'LEY', 'LKY', 'LHY', 'LVY',
                                     'LTR', 'LRES', 'LXAL', 'LYKA', 'LVYKA', 'LS', 'LKYC', 'LKZC', 'LMUV', 'LMX',
                                     'LMY', 'LVMX', 'PCX1', 'PDX1', 'PDX2', 'PDX3', 'PEX1', 'PEX2', 'PEX3',
                                     'PEX4', 'PKX1', 'PKX2', 'PKX3', 'PHX1', 'PHX2', 'PVX1', 'PVX2', 'PPX1',
                                     'PPX2', 'PPX3', 'PPX4', 'RBX1', 'RBX2', 'RBX3', 'RCX1', 'REX1', 'REX2',
                                     'RHX1', 'QSX1', 'QSX2', 'QSX3', 'QSX4', 'QSX5', 'QSX6', 'QSX7', 'QSX8',
                                     'QSX9', 'QSX10', 'QSX11', 'QSX12', 'QSX13', 'QSX14', 'QPMX1', 'PCY1',
                                     'PDY1', 'PDY2', 'PDY3', 'PEY1', 'PEY2', 'PEY3', 'PEY4', 'PEY5', 'PKY1',
                                     'PKY2', 'PKY3', 'PKY4', 'PKY5', 'PKY6', 'PKY7', 'PHY1', 'PHY2', 'PVY1',
                                     'PVY2', 'PVY3', 'PVY4', 'PPY1', 'PPY2', 'PPY3', 'PPY4', 'PPY5', 'RBY1',
                                     'RBY2', 'RBY3', 'RBY4', 'RCY1', 'REY1', 'REY2', 'RHY1', 'RHY2', 'RVY1',
                                     'RVY2', 'RVY3', 'RVY4', 'RVY5', 'RVY6', 'QBZ1', 'QBZ2', 'QBZ3', 'QBZ4',
                                     'QBZ5', 'QBZ6', 'QBZ9', 'QBZ10', 'QCZ1', 'QDZ1', 'QDZ2', 'QDZ3', 'QDZ4',
                                     'QDZ6', 'QDZ7', 'QDZ8', 'QDZ9', 'QDZ10', 'QDZ11', 'QEZ1', 'QEZ2', 'QEZ3',
                                     'QEZ4', 'QEZ5', 'QHZ1', 'QHZ2', 'QHZ3', 'QHZ4', 'PPZ1', 'PPZ2', 'SSZ1',
                                     'SSZ2', 'SSZ3', 'SSZ4']


@dataclass
class Mf61:
    __slots__ = ['FNOMIN', 'UNLOADED_RADIUS', 'LONGVL', 'NOMPRES', 'LFZ0', 'LCX', 'LMUX',
                 'LEX', 'LKX', 'LHX', 'LVX', 'LCY', 'LMUY', 'LEY', 'LKY', 'LHY', 'LVY',
                 'LTR', 'LRES', 'LXAL', 'LYKA', 'LVYKA', 'LS', 'LKYC', 'LKZC', 'LMUV', 'LMX',
                 'LMY', 'LVMX', 'PCX1', 'PDX1', 'PDX2', 'PDX3', 'PEX1', 'PEX2', 'PEX3',
                 'PEX4', 'PKX1', 'PKX2', 'PKX3', 'PHX1', 'PHX2', 'PVX1', 'PVX2', 'PPX1',
                 'PPX2', 'PPX3', 'PPX4', 'RBX1', 'RBX2', 'RBX3', 'RCX1', 'REX1', 'REX2',
                 'RHX1', 'QSX1', 'QSX2', 'QSX3', 'QSX4', 'QSX5', 'QSX6', 'QSX7', 'QSX8',
                 'QSX9', 'QSX10', 'QSX11', 'QSX12', 'QSX13', 'QSX14', 'QPMX1', 'PCY1',
                 'PDY1', 'PDY2', 'PDY3', 'PEY1', 'PEY2', 'PEY3', 'PEY4', 'PEY5', 'PKY1',
                 'PKY2', 'PKY3', 'PKY4', 'PKY5', 'PKY6', 'PKY7', 'PHY1', 'PHY2', 'PVY1',
                 'PVY2', 'PVY3', 'PVY4', 'PPY1', 'PPY2', 'PPY3', 'PPY4', 'PPY5', 'RBY1',
                 'RBY2', 'RBY3', 'RBY4', 'RCY1', 'REY1', 'REY2', 'RHY1', 'RHY2', 'RVY1',
                 'RVY2', 'RVY3', 'RVY4', 'RVY5', 'RVY6', 'QBZ1', 'QBZ2', 'QBZ3', 'QBZ4',
                 'QBZ5', 'QBZ6', 'QBZ9', 'QBZ10', 'QCZ1', 'QDZ1', 'QDZ2', 'QDZ3', 'QDZ4',
                 'QDZ6', 'QDZ7', 'QDZ8', 'QDZ9', 'QDZ10', 'QDZ11', 'QEZ1', 'QEZ2', 'QEZ3',
                 'QEZ4', 'QEZ5', 'QHZ1', 'QHZ2', 'QHZ3', 'QHZ4', 'PPZ1', 'PPZ2', 'SSZ1',
                 'SSZ2', 'SSZ3', 'SSZ4', 'FITTYP', 'WIDTH', 'ASPECT_RATIO', 'VERTICAL_STIFFNESS',
                 'PRESMIN', 'PRESMAX', 'KPUMIN', 'KPUMAX', 'APLMIN', 'ALPMAX', 'CAMMIN', 'CAMMAX']

    # Initializations
    FNOMIN: float
    NOMPRES: float
    UNLOADED_RADIUS: float

    # General Scaling Factors
    LFZ0: float
    LCX: float
    LMUX: float
    LEX: float
    LKX: float
    LHX: float
    LVX: float
    LCY: float
    LMUY: float
    LEY: float
    LKY: float
    LHY: float
    LVY: float
    LTR: float
    LRES: float
    LXAL: float
    LYKA: float
    LVYK: float
    LS: float
    LKYC: float
    LKZ: float
    LMU: float
    LMX: float
    LMY: float
    LVMX: float

    # Longitudinal Coefficients
    PCX1: float
    PDX1: float
    PDX2: float
    PDX3: float
    PEX1: float
    PEX2: float
    PEX3: float
    PEX4: float
    PKX1: float
    PKX2: float
    PKX3: float
    PHX1: float
    PHX2: float
    PVX1: float
    PVX2: float
    PPX1: float
    PPX2: float
    PPX3: float
    PPX4: float
    RBX1: float
    RBX2: float
    RBX3: float
    RCX1: float
    REX1: float
    REX2: float
    RHX1: float

    # Overturning Coefficients
    SX1: float
    QSX2: float
    QSX3: float
    QSX4: float
    QSX5: float
    QSX6: float
    QSX7: float
    QSX8: float
    QSX9: float
    QSX10: float
    QSX11: float
    QSX12: float
    QSX13: float
    QSX14: float
    QPMX1: float

    # Lateral Coefficients
    PCY1: float
    PDY1: float
    PDY2: float
    PDY3: float
    PEY1: float
    PEY2: float
    PEY3: float
    PEY4: float
    PEY5: float
    PKY1: float
    PKY2: float
    PKY3: float
    PKY4: float
    PKY5: float
    PKY6: float
    PKY7: float
    PHY1: float
    PHY2: float
    PVY1: float
    PVY2: float
    PVY3: float
    PVY4: float
    PPY1: float
    PPY2: float
    PPY3: float
    PPY4: float
    PPY5: float
    RBY1: float
    RBY2: float
    RBY3: float
    RBY4: float
    RCY1: float
    REY1: float
    REY2: float
    RHY1: float
    RHY2: float
    RVY1: float
    RVY2: float
    RVY3: float
    RVY4: float
    RVY5: float
    RVY6: float

    # Aligning Moment Coefficients
    QBZ1: float
    QBZ2: float
    QBZ3: float
    QBZ4: float
    QBZ5: float
    QBZ6: float
    QBZ9: float
    QBZ10: float
    QCZ1: float
    QDZ1: float
    QDZ2: float
    QDZ3: float
    QDZ4: float
    QDZ6: float
    QDZ7: float
    QDZ8: float
    QDZ9: float
    QDZ10: float
    QDZ11: float
    QEZ1: float
    QEZ2: float
    QEZ3: float
    QEZ4: float
    QEZ5: float
    QHZ1: float
    QHZ2: float
    QHZ3: float
    QHZ4: float
    PPZ1: float
    PPZ2: float
    SSZ1: float
    SSZ2: float
    SSZ3: float
    SSZ4: float


if __name__ == "__main__":
    qt_helper = Dialogs(__file__ + 'get TIR')
    filename = qt_helper.select_file_dialog(accepted_file_types='*.txt')
    PacejkaMf61(filename)

