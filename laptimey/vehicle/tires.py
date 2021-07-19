"""
This file contains the tire model classes and their functions for the calculation of tire forces

- alpha = slip angle [deg]
- kappa = slip ratio [-]
- gamma = camber [deg]

Created 2021, Contributors: Nigel Swab
"""
import numpy as np

from lib.pyqt_helper import Dialogs


class PacejkaMf61:
    
    def __init__(self, filename):
        # Initializations
        self.FNOMIN: float = 0.0
        self.NOMPRES: float = 0.0
        self.UNLOADED_RADIUS: float = 0.0

        # General Scaling Factors
        self.LFZ0: float = 0.0
        self.LCX: float = 0.0
        self.LMUX: float = 0.0
        self.LEX: float = 0.0
        self.LKX: float = 0.0
        self.LHX: float = 0.0
        self.LVX: float = 0.0
        self.LCY: float = 0.0
        self.LMUY: float = 0.0
        self.LEY: float = 0.0
        self.LKY: float = 0.0
        self.LHY: float = 0.0
        self.LVY: float = 0.0
        self.LTR: float = 0.0
        self.LRES: float = 0.0
        self.LXAL: float = 0.0
        self.LYKA: float = 0.0
        self.LVYK: float = 0.0
        self.LS: float = 0.0
        self.LKYC: float = 0.0
        self.LKZ: float = 0.0
        self.LMU: float = 0.0
        self.LMX: float = 0.0
        self.LMY: float = 0.0
        self.LVMX: float = 0.0

        # Longitudinal Coefficients
        self.PCX1: float = 0.0
        self.PDX1: float = 0.0
        self.PDX2: float = 0.0
        self.PDX3: float = 0.0
        self.PEX1: float = 0.0
        self.PEX2: float = 0.0
        self.PEX3: float = 0.0
        self.PEX4: float = 0.0
        self.PKX1: float = 0.0
        self.PKX2: float = 0.0
        self.PKX3: float = 0.0
        self.PHX1: float = 0.0
        self.PHX2: float = 0.0
        self.PVX1: float = 0.0
        self.PVX2: float = 0.0
        self.PPX1: float = 0.0
        self.PPX2: float = 0.0
        self.PPX3: float = 0.0
        self.PPX4: float = 0.0
        self.RBX1: float = 0.0
        self.RBX2: float = 0.0
        self.RBX3: float = 0.0
        self.RCX1: float = 0.0
        self.REX1: float = 0.0
        self.REX2: float = 0.0
        self.RHX1: float = 0.0

        # Overturning Coefficients
        self.SX1: float = 0.0
        self.QSX2: float = 0.0
        self.QSX3: float = 0.0
        self.QSX4: float = 0.0
        self.QSX5: float = 0.0
        self.QSX6: float = 0.0
        self.QSX7: float = 0.0
        self.QSX8: float = 0.0
        self.QSX9: float = 0.0
        self.QSX10: float = 0.0
        self.QSX11: float = 0.0
        self.QSX12: float = 0.0
        self.QSX13: float = 0.0
        self.QSX14: float = 0.0
        self.QPMX1: float = 0.0

        # Lateral Coefficients
        self.PCY1: float = 0.0
        self.PDY1: float = 0.0
        self.PDY2: float = 0.0
        self.PDY3: float = 0.0
        self.PEY1: float = 0.0
        self.PEY2: float = 0.0
        self.PEY3: float = 0.0
        self.PEY4: float = 0.0
        self.PEY5: float = 0.0
        self.PKY1: float = 0.0
        self.PKY2: float = 0.0
        self.PKY3: float = 0.0
        self.PKY4: float = 0.0
        self.PKY5: float = 0.0
        self.PKY6: float = 0.0
        self.PKY7: float = 0.0
        self.PHY1: float = 0.0
        self.PHY2: float = 0.0
        self.PVY1: float = 0.0
        self.PVY2: float = 0.0
        self.PVY3: float = 0.0
        self.PVY4: float = 0.0
        self.PPY1: float = 0.0
        self.PPY2: float = 0.0
        self.PPY3: float = 0.0
        self.PPY4: float = 0.0
        self.PPY5: float = 0.0
        self.RBY1: float = 0.0
        self.RBY2: float = 0.0
        self.RBY3: float = 0.0
        self.RBY4: float = 0.0
        self.RCY1: float = 0.0
        self.REY1: float = 0.0
        self.REY2: float = 0.0
        self.RHY1: float = 0.0
        self.RHY2: float = 0.0
        self.RVY1: float = 0.0
        self.RVY2: float = 0.0
        self.RVY3: float = 0.0
        self.RVY4: float = 0.0
        self.RVY5: float = 0.0
        self.RVY6: float = 0.0

        # Aligning Moment Coefficients
        self.QBZ1: float = 0.0
        self.QBZ2: float = 0.0
        self.QBZ3: float = 0.0
        self.QBZ4: float = 0.0
        self.QBZ5: float = 0.0
        self.QBZ6: float = 0.0
        self.QBZ9: float = 0.0
        self.QBZ10: float = 0.0
        self.QCZ1: float = 0.0
        self.QDZ1: float = 0.0
        self.QDZ2: float = 0.0
        self.QDZ3: float = 0.0
        self.QDZ4: float = 0.0
        self.QDZ6: float = 0.0
        self.QDZ7: float = 0.0
        self.QDZ8: float = 0.0
        self.QDZ9: float = 0.0
        self.QDZ10: float = 0.0
        self.QDZ11: float = 0.0
        self.QEZ1: float = 0.0
        self.QEZ2: float = 0.0
        self.QEZ3: float = 0.0
        self.QEZ4: float = 0.0
        self.QEZ5: float = 0.0
        self.QHZ1: float = 0.0
        self.QHZ2: float = 0.0
        self.QHZ3: float = 0.0
        self.QHZ4: float = 0.0
        self.PPZ1: float = 0.0
        self.PPZ2: float = 0.0
        self.SSZ1: float = 0.0
        self.SSZ2: float = 0.0
        self.SSZ3: float = 0.0
        self.SSZ4: float = 0.0

        # List of parameters to pull from .TIR files
        self.list_of_model_info = []
        self.list_of_constraints = []
        self.list_of_coefficients = []

        # Dictionaries to store tire model data
        self.model_info = dict()
        self.constraints = dict()
        self.coefficient_descriptions = dict()

        # Load tire model data
        self._load_tire_model_metadata(self)
        self._load_tire_model(self, filename)

        # Initializations
        self.Fz0 = self.FNOMIN * self.LFZ0

    def f_y(self, alpha, kappa, gamma, Fz, tire_pressure):
        """
        Calculates and returns the lateral force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        camber (gamma), normal force  (Fz), and tire pressure
        """

        # Initializations
        dfz = (Fz - self.Fz0) / self.Fz0
        dpi = (tire_pressure - self.NOMPRES) / self.NOMPRES

        # Pure slip
        SVyg = Fz * (self.PVY3 + self.PVY4 * dfz) * gamma * self.LKY * self.LMUY
        SVy0 = Fz * (self.PVY1 + self.PVY2 * dfz) * self.LVY * self.LMUY
        SVy = SVy0 + SVyg
        Kyg = (self.PKY6 + self.PKY7 * dfz) * (1 + self.PPY5 * dpi) * Fz * self.LKYC
        Kya = (1 - self.PKY3 * abs(gamma)) * self.PKY1 * self.Fz0 * (1 + self.PPY1 * dpi) * \
               np.sin(self.PKY4 * np.arctan(Fz / ((self.PKY2 + self.PKY5 * gamma ** 2)
                                                  * (1 + self.PPY2 * dpi) * self.Fz0))) * self.LKY
        SHy0 = (self.PHY1 + self.PHY2 * dfz) * self.LHY
        SHyg = (Kyg * gamma - SVyg) / Kya
        SHy = SHy0 + SHyg
        alpha_y = alpha + SHy
        Ey = (self.PEY1 + self.PEY2 * dfz) * \
             (1 + self.PEY5 * gamma**2 - (self.PEY3 + self.PEY4 * gamma) * np.sign(alpha_y)) * self.LEY
        Mewy = (self.PDY1 + self.PDY2 * dfz) * (1 - self.PDY3 * gamma ** 2) * \
               (1 + self.PPY3 * dpi + self.PPY4 * dpi ** 2) * self.LMUY
        Dy = Mewy * Fz
        Cy = self.PCY1 * self.LCY
        By = Kya / (Cy * Dy)
        SVyk = 0
        Gyk = 1

        # Combined Slip (if there is a slip ratio
        if kappa:
            SHyk = self.RHY1 + self.RHY2 * dfz
            Eyk = self.REY1 + self.REY2 * dfz
            Cyk = self.RCY1
            Byk = (self.RBY1 + self.RBY4 * gamma ** 2) * \
                  np.cos( np.arctan(self.RBY2 * (alpha - self.RBY3))) * self.LYKA
            kappa_s = kappa + SHyk
            Gyk = (np.cos(Cyk * np.arctan(Byk * kappa_s - Eyk * (Byk * kappa_s - np.arctan(Byk * kappa_s))))) / \
                  (np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk)))))
            DVyk = Mewy * Fz * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma) * \
                   np.cos(np.arctan(self.RVY4 * alpha))
            SVyk = DVyk * np.sin(self.RVY5 * np.arctan(self.RVY6)) * self.LVYK

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
        dfz = (Fz - self.Fz0) / self.Fz0
        dpi = (tire_pressure - self.NOMPRES) / self.NOMPRES

        # Pure Slip
        SVx = (self.PVX1 + self.PVX2 * dfz) * Fz * self.LVX * self.LMUX
        SHx = (self.PHX1 + self.PHX2 * dfz) * self.LHX
        kappa_x = kappa + SHx
        Kxk = (self.PKX1 + self.PKX2 * dfz) * np.exp(self.PKX3 * dfz) * \
              (1 + self.PPX1 * dpi + self.PPX2 * dpi ** 2) * Fz * self.LKX
        Ex = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz ** 2) * \
             (1 - self.PEX4 * np.sign(kappa_x)) * self.LEX
        Mewx = (self.PDX1 + self.PDX2 * dfz) * (1 - self.PDX3 * gamma ** 2) * \
               (1 + self.PPX3 * dpi + self.PPX4 * dpi ** 2) * Fz * self.LMUX
        Dx = Mewx * Fz
        Cx = self.PCX1 * self.LCX
        Bx = Kxk / (Cx * Dx)
        Gxa = 1

        # Combined Slip
        if alpha:
            SHxa = self.RHX1
            Exa = self.REX1 + self.REX2 * dfz
            Cxa = self.RCX1
            Bxa = (self.RBX1 + self.RBX3 * gamma ** 2) * np.cos(np.arctan(self.RBX2 * kappa)) * self.LXAL
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
        dfz = (Fz - self.Fz0) / self.Fz0
        dpi = (tire_pressure - self.NOMPRES) / self.NOMPRES



        # Residual Moment (Mzr)
        Dr = ((self.QDZ6 + self.QDZ7 * dfz) * self.LRES + (self.QDZ8 + self.QDZ9 * dfz) *
              (1 - self.PPZ2 * dpi) * gamma * self.LKZ + (self.QDZ10 + self.QDZ11 * dfz) *
              gamma * np.abs(gamma) * self.LKZ) * Fz * self.UNLOADED_RADIUS * self.LMUY
        Br = self.QBZ9 * self.LKY / self.LMUY + self.QBZ10 * By * Cy
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


if __name__ == "__main__":
    qt_helper = Dialogs(__file__ + 'get TIR')
    filename = qt_helper.select_file_dialog(accepted_file_types='*.txt')
    PacejkaMf61(filename)

