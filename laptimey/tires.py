"""
This file contains the tire model classes and their functions for the calculation of tire forces.
All calculations are done in the ISO coordinate frame for tires.

- alpha = slip angle [deg]
- kappa = slip ratio [-]
- gamma = inclination_angle [deg]



For converting TIR models,
- ENSURE THAT TIRE FILE HAS A FITTYPE = XX (52, or 61)
-- Not inherently included in Mf52 optimumT outputs

Assumptions:
- Constant radius used
- Corrections for large inclination_angle angles not used for Mz
- Steady state operation
- Constant tire temperature
- More in specific models (still need to add)
- So many, need to come back to list all of them lol

Created: 2021
Contributors: Nigel Swab
"""
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as df

from lib.pyqt_helper import Dialogs


# TODO: (Maybe) add provisions for "low speed"


class MagicFormula(ABC):

    __slots__ = [
        'MODEL', 'FNOMIN', 'UNLOADED_RADIUS', 'LONGVL', 'NOMPRES', 'Fz0',
        'LFZ0', 'LCX', 'LMUX', 'LEX', 'LKX', 'LHX', 'LVX', 'LCY', 'LMUY',
        'LEY', 'LKY', 'LHY', 'LVY', 'LTR', 'LRES', 'LXAL', 'LYKA', 'LVYKA',
        'LS', 'LKYC', 'LKZC', 'LMUV', 'LMX', 'LMY', 'LVMX', 'PCX1', 'PDX1',
        'PDX2', 'PDX3', 'PEX1', 'PEX2', 'PEX3', 'PEX4', 'PKX1', 'PKX2', 'PKX3',
        'PHX1', 'PHX2', 'PVX1', 'PVX2', 'PPX1', 'PPX2', 'PPX3', 'PPX4', 'RBX1',
        'RBX2', 'RBX3', 'RCX1', 'REX1', 'REX2', 'PTX1', 'PTX2', 'PTX3', 'RHX1',
        'QSX1', 'QSX2', 'QSX3', 'QSX4', 'QSX5', 'QSX6', 'QSX7', 'QSX8', 'QSX9',
        'QSX10', 'QSX11', 'QSX12', 'QSX13', 'QSX14', 'QPMX1', 'PCY1', 'PDY1',
        'PDY2', 'PDY3', 'PEY1', 'PEY2', 'PEY3', 'PEY4', 'PEY5', 'PKY1', 'PKY2',
        'PKY3', 'PKY4', 'PKY5', 'PKY6', 'PKY7', 'PHY1', 'PHY2', 'PHY3', 'PVY1',
        'PVY2', 'PVY3', 'PVY4', 'PPY1', 'PPY2', 'PPY3', 'PPY4', 'PPY5', 'RBY1',
        'RBY2', 'RBY3', 'RBY4', 'RCY1', 'REY1', 'REY2', 'RHY1', 'RHY2', 'RVY1',
        'RVY2', 'RVY3', 'RVY4', 'RVY5', 'RVY6', 'PTY1', 'PTY2', 'QBZ1', 'QBZ2',
        'QBZ3', 'QBZ4', 'QBZ5', 'QBZ6', 'QBZ9', 'QBZ10', 'QCZ1', 'QDZ1',
        'QDZ2', 'QDZ3', 'QDZ4', 'QDZ6', 'QDZ7', 'QDZ8', 'QDZ9', 'QDZ10',
        'QDZ11', 'QEZ1', 'QEZ2', 'QEZ3', 'QEZ4', 'QEZ5', 'QHZ1', 'QHZ2',
        'QHZ3', 'QHZ4', 'PPZ1', 'PPZ2', 'SSZ1', 'SSZ2', 'SSZ3', 'SSZ4', 'QTZ1',
        'FITTYP', 'WIDTH', 'RIM_WIDTH', 'RIM_RADIUS', 'VERTICAL_STIFFNESS',
        'PRESMIN', 'PRESMAX', 'KPUMIN', 'KPUMAX', 'ALPMIN', 'ALPMAX', 'CAMMIN',
        'CAMMAX'
    ]

    # Model Information and Dimensions
    MODEL: float  # Expected model to be used when passed in
    FITTYP: float  # Magic formula number (ie. 61 = MF6.1)
    WIDTH: float  # Free tyre radius
    RIM_RADIUS: float  # Nominal aspect ratio
    RIM_WIDTH: float  # Nominal rim radius
    VERTICAL_STIFFNESS: float  # Tire Vertical Stiffness
    PRESMIN: float  # Minimum Valid Tire Pressure (Pa)
    PRESMAX: float  # Maximum Valid Tire Pressure (Pa)
    KPUMIN: float  # Minimum Valid Slip Ratio (-)
    KPUMAX: float  # Maximum Valid Slip Ratio (-)
    ALPMIN: float  # Minimum Valid Slip Angle (rad)
    ALPMAX: float  # Maximum Valid Slip Angle (rad)
    CAMMIN: float  # Minimum Valid inclination_angle Angle (rad)
    CAMMAX: float  # Maximum Valid inclination_angle Angle (rad)

    # Initializations
    FNOMIN: float  # Nominal Vertical Load
    UNLOADED_RADIUS: float  # Unloaded Radius
    LONGVL: float  # Nominal Speed
    NOMPRES: float  # Nominal Pressure
    Fz0: float  # Nominal Vertical Load with scaling factor LFZ0

    # General Scaling Factors
    LFZ0: float  # Scale factor of nominal (rated) load
    LCX: float  # Scale factor of Fx shape factor
    LMUX: float  # Scale factor of Fx peak friction coefficient
    LEX: float  # Scale factor of Fx curvature factor
    LKX: float  # Scale factor of Fx slip stiffness
    LHX: float  # Scale factor of Fx horizontal shift
    LVX: float  # Scale factor of Fx vertical shift
    LCY: float  # Scale factor of Fy shape factor
    LMUY: float  # Scale factor of Fy peak friction coefficient
    LEY: float  # Scale factor of Fy curvature factor
    LKY: float  # Scale factor of Fy cornering stiffness
    LHY: float  # Scale factor of Fy horizontal shift
    LVY: float  # Scale factor of Fy vertical shift
    LTR: float  # Scale factor of peak of pneumatic trail
    LRES: float  # Scale factor for offset of residual torque
    LXAL: float  # Scale factor of alpha influence on Fx
    LYKA: float  # Scale factor of alpha influence on Fx
    LVYKA: float  # Scale factor of kappa induced Fy
    LS: float  # Scale factor of moment arm of Fx
    LKYC: float  # Scale factor of inclination_angle force stiffness
    LKZC: float  # Scale factor of inclination_angle torque stiffness
    LMUV: float  # Scale factor with slip speed decaying friction
    LMX: float  # Scale factor of overturning couple
    LMY: float  # Scale factor of rolling resistance torque
    LVMX: float  # Overturning couple vertical shift

    # Longitudinal Coefficients
    PCX1: float  # Shape factor Cfx for longitudinal force
    PDX1: float  # Longitudinal friction Mux at Fznom
    PDX2: float  # Variation of friction Mux with load
    PDX3: float  # Variation of friction Mux with inclination_angle squared
    PEX1: float  # Longitudinal curvature Efx at Fznom
    PEX2: float  # Variation of curvature Efx with load
    PEX3: float  # Variation of curvature Efx with load squared
    PEX4: float  # Factor in curvature Efx while driving
    PKX1: float  # Longitudinal slip stiffness Kfx/Fz at Fznom
    PKX2: float  # Variation of slip stiffness Kfx/Fz with load
    PKX3: float  # Exponent in slip stiffness Kfx/Fz with load
    PHX1: float  # Horizontal shift Shx at Fznom
    PHX2: float  # Variation of shift Shx with load
    PVX1: float  # Vertical shift Svx/Fz at Fznom
    PVX2: float  # Variation of shift Svx/Fz with load
    PPX1: float  # linear influence of inflation pressure on longitudinal slip stiffness
    PPX2: float  # quadratic influence of inflation pressure on longitudinal slip stiffness
    PPX3: float  # linear influence of inflation pressure on peak longitudinal friction
    PPX4: float  # quadratic influence of inflation pressure on peak longitudinal friction
    RBX1: float  # Slope factor for combined slip Fx reduction
    RBX2: float  # Variation of slope Fx reduction with kappa
    RBX3: float  # Influence of inclination_angle on stiffness for Fx combined
    RCX1: float  # Shape factor for combined slip Fx reduction
    REX1: float  # Curvature factor of combined Fx
    REX2: float  # Curvature factor of combined Fx with load
    RHX1: float  # Shift factor for combined slip Fx reduction
    PTX1: float  # Relaxation length SigKap0/Fz at Fznom
    PTX2: float  # Variation of SigKap0/Fz with load
    PTX3: float  # Variation of SigKap0/Fz with exponent

    # Overturning Coefficients
    SX1: float  # Vertical shift of overturning moment
    QSX2: float  # inclination_angle induced overturning couple
    QSX3: float  # Fy induced overturning couple
    QSX4: float  # Mixed load lateral force and inclination_angle on Mx
    QSX5: float  # Load effect on Mx with lateral force and inclination_angle
    QSX6: float  # B-factor of load with Mx
    QSX7: float  # inclination_angle with load on Mx
    QSX8: float  # Lateral force with load on Mx
    QSX9: float  # B-factor of lateral force with load on Mx
    QSX10: float  # Vertical force with inclination_angle on Mx
    QSX11: float  # B-factor of vertical force with inclination_angle on Mx
    QSX12: float  # inclination_angle squared induced overturning moment
    QSX13: float  # Lateral force induced overturning moment
    QSX14: float  # Lateral force induced overturning moment with inclination_angle
    QPMX1: float  # Influence of inflation pressure on overturning moment

    # Lateral Coefficients
    PCY1: float  # Shape factor Cfy for lateral forces
    PDY1: float  # Lateral friction Muy
    PDY2: float  # Variation of friction Muy with load
    PDY3: float  # Variation of friction Muy with squared inclination_angle
    PEY1: float  # Lateral curvature Efy at Fznom
    PEY2: float  # Variation of curvature Efy with load
    PEY3: float  # Zero order inclination_angle dependency of curvature Efy
    PEY4: float  # Variation of curvature Efy with inclination_angle
    PEY5: float  # Variation of curvature Efy with inclination_angle squared
    PKY1: float  # Maximum value of stiffness Kfy/Fznom
    PKY2: float  # Load at which Kfy reaches maximum value
    PKY3: float  # Variation of Kfy/Fznom with inclination_angle
    PKY4: float  # Curvature of stiffness Kfy
    PKY5: float  # Peak stiffness variation with inclination_angle squared
    PKY6: float  # Fy inclination_angle stiffness factor
    PKY7: float  # Vertical load dependency of inclination_angle stiffness
    PHY1: float  # Horizontal shift Shy at Fznom
    PHY2: float  # Variation of shift Shy with load
    PVY1: float  # Vertical shift in Svy/Fz at Fznom
    PVY2: float  # Variation of shift Svy/Fz with load
    PVY3: float  # Variation of shift Svy/Fz with inclination_angle
    PVY4: float  # Variation of shift Svy/Fz with inclination_angle and load
    PPY1: float  # influence of inflation pressure on cornering stiffness
    PPY2: float  # influence of inflation pressure on dependency of nominal tyre load on cornering stiffness
    PPY3: float  # linear influence of inflation pressure on lateral peak friction
    PPY4: float  # quadratic influence of inflation pressure on lateral peak friction
    PPY5: float  # Influence of inflation pressure on inclination_angle stiffness
    RBY1: float  # Slope factor for combined Fy reduction
    RBY2: float  # Variation of slope Fy reduction with alpha
    RBY3: float  # Shift term for alpha in slope Fy reduction
    RBY4: float  # Influence of inclination_angle on stiffness of Fy combined
    RCY1: float  # Shape factor for combined Fy reduction
    REY1: float  # Curvature factor of combined Fy
    REY2: float  # Curvature factor of combined Fy with load
    RHY1: float  # Shift factor for combined Fy reduction
    RHY2: float  # Shift factor for combined Fy reduction with load
    RVY1: float  # Kappa induced side force Svyk/Muy*Fz at Fznom
    RVY2: float  # Variation of Svyk/Muy*Fz with load
    RVY3: float  # Variation of Svyk/Muy*Fz with inclination_angle
    RVY4: float  # Variation of Svyk/Muy*Fz with alpha
    RVY5: float  # Variation of Svyk/Muy*Fz with kappa
    RVY6: float  # Variation of Svyk/Muy*Fz with atan(kappa)

    # Aligning Moment Coefficients
    QBZ1: float  # Trail slope factor for trail Bpt at Fznom
    QBZ2: float  # Variation of slope Bpt with load
    QBZ3: float  # Variation of slope Bpt with load squared
    QBZ4: float  # Variation of slope Bpt with inclination_angle
    QBZ5: float  # Variation of slope Bpt with absolute inclination_angle
    QBZ6: float  # inclination_angle influence Bt
    QBZ9: float  # Factor for scaling factors of slope factor Br of Mzr
    QBZ10: float  # Factor for dimensionless cornering stiffness of Br of Mzr
    QCZ1: float  # Shape factor Cpt for pneumatic trail
    QDZ1: float  # Peak trail Dpt = Dpt*(Fz/Fznom*R0)
    QDZ2: float  # Variation of peak Dpt" with load
    QDZ3: float  # Variation of peak Dpt" with inclination_angle
    QDZ4: float  # Variation of peak Dpt" with inclination_angle squared
    QDZ6: float  # Peak residual torque Dmr" = Dmr/(Fz*R0)
    QDZ7: float  # Variation of peak factor Dmr" with load
    QDZ8: float  # Variation of peak factor Dmr" with inclination_angle
    QDZ9: float  # Variation of peak factor Dmr" with inclination_angle and load
    QDZ10: float  # Variation of peak factor Dmr with inclination_angle squared
    QDZ11: float  # Variation of Dmr with inclination_angle squared and load
    QEZ1: float  # Trail curvature Ept at Fznom
    QEZ2: float  # Variation of curvature Ept with load
    QEZ3: float  # Variation of curvature Ept with load squared
    QEZ4: float  # Variation of curvature Ept with sign of Alpha-t
    QEZ5: float  # Variation of Ept with inclination_angle and sign Alpha-t
    QHZ1: float  # Trail horizontal shift Sht at Fznom
    QHZ2: float  # Variation of shift Sht with load
    QHZ3: float  # Variation of shift Sht with inclination_angle
    QHZ4: float  # Variation of shift Sht with inclination_angle and load
    PPZ1: float  # effect of inflation pressure on length of pneumatic trail
    PPZ2: float  # Influence of inflation pressure on residual aligning torque
    SSZ1: float  # Nominal value of s/R0: effect of Fx on Mz
    SSZ2: float  # Variation of distance s/R0 with Fy/Fznom
    SSZ3: float  # Variation of distance s/R0 with inclination_angle
    SSZ4: float  # Variation of distance s/R0 with load and inclination_angle

    def load_model_from_tir(self, filepath: str) -> None:
        ''' Takes the passed in file path, extracts the necessary data, and populates the dictionary with both the key
        and data values.
        '''

        print(f'Loading tire model from {filepath}')

        # Load all lines of data
        with open(filepath, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Remove whitespaces
            line = "".join(line.split())

            # Ensure it's not a header or divider line
            if line and line[0].isalpha():
                # Separate the parameter key and it's value
                key, value = line.split(sep='=', maxsplit=1)
                # Check if the key is an attribute, assign the value to it if it is
                if key in self.__slots__:
                    value = value.split(sep='$', maxsplit=1)
                    setattr(self, key, float(value[0]))
        self.Fz0 = self.FNOMIN * self.LFZ0

        if self.MODEL != self.FITTYP:
            raise ValueError(
                f'Expecting Mf{self.MODEL: .0f}, got Mf{self.FITTYP: .0f}')
        else:
            print('Tire model loaded \n')

    def create_tire_plot_test_data(self, alpha_range, kappa_range, gamma, Fz,
                                   tire_pressure):

        # Initialize lists for forces
        fy_forces = []
        fx_forces = []
        mz_forces = []
        mx_forces = []

        kappa = 0
        for alpha in alpha_range:
            Fy, _, Mz, Mx = self.tire_forces(alpha, kappa, gamma, Fz,
                                             tire_pressure)
            fy_forces.append(Fy)
            mz_forces.append(Mz)
            mx_forces.append(Mx)

        alpha = 0
        for kappa in kappa_range:
            _, Fx, _, _ = self.tire_forces(alpha, kappa, gamma, Fz,
                                           tire_pressure)
            fx_forces.append(Fx)
        alpha_range = np.rad2deg(alpha_range)
        headings = [
            'Slip Angle [deg]', 'Slip Ratio [-]', 'Fy [N]', 'Fx [N]',
            'Mz [Nm]', 'Mx [Nm]'
        ]
        return df(
            data=list(
                zip(
                    alpha_range,
                    kappa_range,
                    fy_forces,
                    fx_forces,
                    mz_forces,
                    mx_forces,
                )),
            columns=headings,
        )

    def plot_force(self, data):
        # TODO: - Add plot labels

        fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2)

        # Fy
        slip_angles = data['Slip Angle [deg]']
        fy = data['Fy [N]']
        ax00.plot(
            slip_angles,
            fy,
        )
        ax00.set_xlabel('Slip Angle [deg]')
        ax00.set_ylabel('Fy [N]')
        ax00.set_title('Tire Lateral Force')

        # Fx
        slip_ratios = data['Slip Ratio [-]']
        fx = data['Fx [N]']
        ax01.plot(slip_ratios, fx)
        ax01.set_xlabel('Slip Ratio [-]')
        ax01.set_ylabel('Fx [n]')
        ax01.set_title('Tire Longitudinal Force')

        # Mz
        mz = data['Mz [Nm]']
        ax10.plot(slip_angles, mz)
        ax10.set_xlabel('Slip Angle [deg]')
        ax10.set_ylabel('Mz [Nm]')
        ax10.set_title('Tire Aligning Moment')

        # Mx
        mx = data['Mx [Nm]']
        ax11.plot(slip_angles, mx)
        ax11.set_xlabel('Slip Angle [deg]')
        ax11.set_ylabel('Mx [Nm]')
        ax11.set_title('Tire Overturning Moment')

        fig.suptitle('Tire Forces', fontsize=16)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        mng.set_window_title('Tire Forces')
        plt.subplots_adjust(wspace=0.2, hspace=0.25)
        plt.show()

    def tire_forces(self, alpha, kappa, gamma, Fz, tire_pressure):
        '''

        :param self:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param tire_pressure:
        :return:
        '''

        # return no forces/moments if there is not positive normal force
        if Fz <= 0:
            return 0, 0, 0, 0

        # Initializations
        dfz = (Fz - self.Fz0) / self.Fz0
        dpi = (tire_pressure - self.NOMPRES) / self.NOMPRES

        Fy, By, Cy, SVy, SHy, Kya = self.f_y(alpha, kappa, gamma, Fz, dfz, dpi)
        Fx, Kxk = self.f_x(alpha, kappa, gamma, Fz, dfz, dpi)
        Mz = self.m_z(alpha, kappa, gamma, Fz, Fy, Fx, dfz, dpi, By, Cy, SVy,
                      SHy, Kya, Kxk)
        Mx = self.m_x(gamma, Fz, Fy, dpi)

        return Fy, Fx, Mz, Mx

    @abstractmethod
    def f_y(self, alpha, kappa, gamma, Fz, dfz, dpi):
        pass

    @abstractmethod
    def f_x(self, alpha, kappa, gamma, Fz, dfz, dpi):
        pass

    @abstractmethod
    def m_z(self, alpha, kappa, gamma, Fz, Fy, Fx, dfz, dpi, By, Cy, SVy, SHy,
            Kya, Kxk):
        pass

    @abstractmethod
    def m_x(self, gamma, Fz, Fy, dpi):
        pass


class Mf61(MagicFormula):
    def __init__(self, filepath: str):
        self.MODEL = 61
        self.load_model_from_tir(filepath=filepath)

    def f_y(self, alpha, kappa, gamma, Fz, dfz, dpi):
        '''Calculates and returns the lateral force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        inclination_angle (gamma), normal force  (Fz), and tire pressure

        :param self:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param dfz:
        :param dpi:
        :return:
        '''

        # Pure slip
        SVyg = Fz * (self.PVY3 +
                     self.PVY4 * dfz) * gamma * self.LKYC * self.LMUY
        SVy0 = Fz * (self.PVY1 + self.PVY2 * dfz) * self.LVY * self.LMUY
        SVy = SVy0 + SVyg
        Kyg = (self.PKY6 +
               self.PKY7 * dfz) * (1 + self.PPY5 * dpi) * Fz * self.LKYC
        Kya = (1 - self.PKY3 * abs(gamma)) * self.PKY1 * self.Fz0 * (1 + self.PPY1 * dpi) \
            * np.sin(self.PKY4 * np.arctan(Fz / ((self.PKY2 + self.PKY5 * np.power(gamma, 2))
                                                 * (1 + self.PPY2 * dpi) * self.Fz0))) * self.LKY

        SHy0 = (self.PHY1 + self.PHY2 * dfz) * self.LHY
        SHyg = (Kyg * gamma - SVyg) / Kya
        SHy = SHy0 + SHyg
        alpha_y = alpha + SHy
        Ey = (self.PEY1 + self.PEY2 * dfz) * \
             (1 + self.PEY5 * gamma ** 2 - (self.PEY3 + self.PEY4 * gamma) * np.sign(alpha_y)) * self.LEY
        Mewy = (self.PDY1 + self.PDY2 * dfz) * (1 - self.PDY3 * gamma ** 2) * \
               (1 + self.PPY3 * dpi + self.PPY4 * gamma ** 2) * self.LMUY
        Dy = Mewy * Fz
        Cy = self.PCY1 * self.LCY
        By = Kya / (Cy * Dy)
        SVyk = 0
        Gyk = 1

        # Combined Slip (if there is a slip ratio)
        if kappa:
            SHyk = self.RHY1 + self.RHY2 * dfz
            Eyk = self.REY1 + self.REY2 * dfz
            Cyk = self.RCY1
            Byk = (self.RBY1 + self.RBY4 * gamma ** 2) * \
                  np.cos(np.arctan(self.RBY2 * (alpha - self.RBY3))) * self.LYKA
            kappa_s = kappa + SHyk
            Gyk = (np.cos(Cyk * np.arctan(Byk * kappa_s - Eyk * (Byk * kappa_s - np.arctan(Byk * kappa_s))))) / \
                  (np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk)))))
            DVyk = Mewy * Fz * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma) * \
                   np.cos(np.arctan(self.RVY4 * alpha))
            SVyk = DVyk * np.sin(self.RVY5 * np.arctan(self.RVY6)) * self.LVYKA

        # Pure Lateral Force
        By_x_alpha_y = By * alpha_y
        Fyp = Dy * np.sin(
            Cy * np.arctan(By_x_alpha_y - Ey *
                           (By_x_alpha_y - np.arctan(By_x_alpha_y)))) + SVy

        # Combined Lateral Force ( Fyp == Fy if kappa == 0)
        Fy = Gyk * Fyp + SVyk
        return Fy, By, Cy, SVy, SHy, Kya

    def f_x(self, alpha, kappa, gamma, Fz, dfz, dpi):
        ''' Calculates and returns the longitudinal force produced by a tire given a slip angle (alpha),
        slip ratio (kappa), inclination_angle (gamma), normal force  (Fz), and tire pressure

        :param self:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param dfz:
        :param dpi:
        :return:
        '''

        # Pure Slip
        SVx = (self.PVX1 + self.PVX2 * dfz) * Fz * self.LVX * self.LMUX
        SHx = (self.PHX1 + self.PHX2 * dfz) * self.LHX
        kappa_x = kappa + SHx
        Kxk = (self.PKX1 + self.PKX2 * dfz) * np.exp(self.PKX3 * dfz) * \
              (1 + self.PPX1 * dpi + self.PPX2 * dpi ** 2) * Fz * self.LKX
        Ex = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz ** 2) * \
             (1 - self.PEX4 * np.sign(kappa_x)) * self.LEX
        Mewx = (self.PDX1 + self.PDX2 * dfz) * (1 - self.PDX3 * gamma ** 2) * \
               (1 + self.PPX3 * dpi + self.PPX4 * dpi ** 2) * self.LMUX
        Dx = Mewx * Fz
        Cx = self.PCX1 * self.LCX
        Bx = Kxk / (Cx * Dx)
        # Combined Slip
        Gxa = self.calc_combined_long_force_scalar(dfz, gamma, kappa,
                                                   alpha) if alpha else 1
        # Longitudinal Force (N)
        Bx_kappa_x = Bx * kappa_x
        Fx = (Dx * np.sin(Cx * np.arctan(Bx_kappa_x - Ex *
                                         (Bx_kappa_x - np.arctan(Bx_kappa_x))))
              + SVx) * Gxa
        return Fx, Kxk

    # TODO Rename this here and in `f_x`
    def calc_combined_long_force_scalar(self, dfz, gamma, kappa, alpha):
        SHxa = self.RHX1
        Exa = self.REX1 + self.REX2 * dfz
        Cxa = self.RCX1
        Bxa = (self.RBX1 + self.RBX3 * gamma**2) * np.cos(
            np.arctan(self.RBX2 * kappa)) * self.LXAL
        alpha_s = alpha + SHxa
        Bxa_x_alpha_s = Bxa * alpha_s
        Bxa_x_SHxa = Bxa * SHxa
        return (np.cos(
            Cxa * np.arctan(Bxa_x_alpha_s - Exa *
                            (Bxa_x_alpha_s - np.arctan(Bxa_x_alpha_s)))
        )) / (np.cos(Cxa * np.arctan(Bxa_x_SHxa - Exa *
                                     (Bxa_x_SHxa - np.arctan(Bxa_x_SHxa)))))

    def m_z(self, alpha, kappa, gamma, Fz, Fy, Fx, dfz, dpi, By, Cy, SVy, SHy,
            Kya, Kxk):
        '''
        NOTE: Fy at 0 inclination_angle assumed approximately equal to calculated Fy to save computation time (for now)

        :param self:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param Fy:
        :param Fx:
        :param dfz:
        :param dpi:
        :param By:
        :param Cy:
        :param SVy:
        :param SHy:
        :param Kya:
        :param Kxk:
        :return:
        '''

        # Computational time not worth recalculating Fy with 0 inclination_angle given small inclination_angle angles
        Fyp0 = Fy
        # TODO: Test to see how much of a difference the 0 inclination_angle Fy makes
        # Fyp0, By, Cy, SVy, SHy, Kya = self.f_y(alpha, kappa, gamma, Fz, dfz, dpi)

        # alpha_m = alpha when disregarding transient effects
        alpha_m = alpha
        SHt = self.QHZ1 + self.QHZ2 * dfz + (self.QHZ3 +
                                             self.QHZ4 * dfz) * gamma
        alpha_r = alpha_m + SHy + SVy / Kya
        alpha_t = alpha_m + SHt

        if not kappa or not alpha_m:  # if pure slip
            alpha_teq = alpha_t
            alpha_req = alpha_r
            s = 0
        else:
            alpha_teq = np.arctan(
                np.sqrt((np.tan(alpha_t))**2 +
                        (Kxk / Kya)**2 * kappa**2)) * np.sign(alpha_t)
            alpha_req = np.arctan(
                np.sqrt((np.tan(alpha_r))**2 +
                        (Kxk / Kya)**2 * kappa**2)) * np.sign(alpha_r)
            s = (self.SSZ1 + self.SSZ2 * (Fy / self.Fz0) + (self.SSZ3 + self.SSZ4 * dfz) * gamma) \
                * self.UNLOADED_RADIUS * self.LS

        # Pnneumatic Trail t
        Bt = (self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz ** 2) \
            * (1 + self.QBZ4 * gamma + self.QBZ5 * np.abs(gamma) + self.QBZ6 * gamma ** 2) * self.LKY / self.LMUY
        # Note: Have had odd QBZ4 Values from Optimum tire before, resulting in wonky Mz
        Ct = self.QCZ1
        Dt = Fz * (self.QDZ1 + self.QDZ2 * dfz) * (1 - self.PPZ1 * dpi) \
            * (1 + self.QDZ3 * gamma + self.QDZ4 * gamma ** 2) * self.UNLOADED_RADIUS / self.Fz0 * self.LTR
        Et = (self.QEZ1 + self.QEZ2 * dfz + self.QEZ3 * dfz ** 2) \
            * (1 + (self.QEZ4 + self.QEZ5 * gamma) * (2 / np.pi) * np.arctan(Bt * Ct * alpha_t))
        Bt_x_alpha_teq = Bt * alpha_teq
        t = Dt * np.cos(Ct * np.arctan(Bt_x_alpha_teq - Et * (Bt_x_alpha_teq - np.arctan(Bt_x_alpha_teq)))) \
            * np.cos(alpha_m)

        # Residual Moment (Mzr)
        Dr = ((self.QDZ6 + self.QDZ7 * dfz) * self.LRES +
              (self.QDZ8 + self.QDZ9 * dfz) *
              (1 - self.PPZ2 * dpi) * gamma * self.LKZC +
              (self.QDZ10 + self.QDZ11 * dfz) * gamma * np.abs(gamma) *
              self.LKZC) * Fz * self.UNLOADED_RADIUS * self.LMUY
        Br = self.QBZ9 * self.LKY / self.LMUY + self.QBZ10 * By * Cy
        Mzr = Dr * np.cos(np.arctan(Br * alpha_req)) * np.cos(alpha_m)

        return -t * Fyp0 + Mzr + s * Fx

    def m_x(self, gamma, Fz, Fy, dpi):
        '''

        :param self:
        :param gamma:
        :param Fz:
        :param Fy:
        :param dpi:
        :return:
        '''

        #  Fy normalized by Fz0
        Fy_div_Fz0 = Fy / self.Fz0

        'Not sure why the last term throws everything off, OptimumTire may not fit properly for it?'

        return (
            self.UNLOADED_RADIUS * Fz * self.LMX *
            (self.QSX1 * self.LVMX - self.QSX2 * gamma *
             (1 + self.QPMX1 * dpi) - self.QSX12 * gamma * np.abs(gamma) +
             self.QSX3 * Fy_div_Fz0 + self.QSX4 * np.cos(self.QSX5 * np.arctan(
                 (self.QSX6 * Fz / self.Fz0)**2)) *
             np.sin(self.QSX7 * gamma +
                    self.QSX8 * np.arctan(self.QSX9 * Fy_div_Fz0)) +
             self.QSX10 * np.arctan(self.QSX11 * Fy_div_Fz0) * gamma))


class Mf52(MagicFormula):
    """Equations from https://drive.google.com/file/d/1qjyM6F8YzKPEFXYUvE8Ptty8vhaR5SVw/view?usp=sharing"""
    def __init__(self, filepath: str):
        self.MODEL = 52
        self.load_model_from_tir(filepath=filepath)

    def f_y(self, alpha, kappa, gamma, Fz, dfz, dpi):
        '''Calculates and returns the lateral force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        inclination_angle (gamma), normal force  (Fz), and tire pressure

        :param self:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param dfz:
        :param dpi:
        :return:
        '''

        # Pure slip
        SVy0 = Fz * (self.PVY1 + self.PVY2 * dfz) * self.LVY * self.LMUY
        SVyg = Fz * (self.PVY3 +
                     self.PVY4 * dfz) * gamma * self.LKYC * self.LMUY
        SVy = SVy0 + SVyg
        SHy = (self.PHY1 + self.PHY2 * dfz) * self.LHY + self.PHY3 * gamma
        alpha_y = alpha + SHy
        Kya = self.PKY1 * self.Fz0 * np.sin(np.arctan(Fz / (self.PKY2 * self.Fz0))) \
            * (1 - self.PKY3 * np.abs(gamma)) * self.LKY
        Ey = (self.PEY1 + self.PEY2 * dfz) * (
            1 - (self.PEY3 + self.PEY4 * gamma) * np.sign(alpha_y)) * self.LEY
        Mewy = (self.PDY1 +
                self.PDY2 * dfz) * (1 - self.PDY3 * gamma**2) * self.LMUY
        Dy = Mewy * Fz
        Cy = self.PCY1 * self.LCY
        By = Kya / (Cy * Dy)
        SVyk = 0
        Gyk = 1

        # Combined Slip (if there is a slip ratio)
        if kappa:
            SHyk = self.RHY1 + self.RHY2 * dfz
            Eyk = self.REY1 + self.REY2 * dfz
            Cyk = self.RCY1
            Byk = self.RBY1 * np.cos(np.arctan(
                self.RBY2 * (alpha - self.RBY3))) * self.LYKA
            kappa_s = kappa + SHyk
            Gyk = (np.cos(Cyk * np.arctan(Byk * kappa_s - Eyk * (Byk * kappa_s - np.arctan(Byk * kappa_s))))) / \
                  (np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk)))))
            DVyk = Mewy * Fz * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma) * \
                   np.cos(np.arctan(self.RVY4 * alpha))
            SVyk = DVyk * np.sin(self.RVY5 * np.arctan(self.RVY6)) * self.LVYKA

        # Pure Lateral Force
        Fyp = Dy * np.sin(
            Cy * np.arctan(By * alpha_y - Ey *
                           (By * alpha_y - np.arctan(By * alpha_y)))) + SVy

        # Combined Lateral Force ( Fyp == Fy if kappa == 0)
        Fy = Gyk * Fyp + SVyk

        return Fy, By, Cy, SVy, SHy, Kya

    def f_x(self, alpha, kappa, gamma, Fz, dfz, dpi):
        ''' Calculates and returns the longitudinal force produced by a tire given a slip angle (alpha),
        slip ratio (kappa), inclination_angle (gamma), normal force  (Fz), and tire pressure

        :param self:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param dfz:
        :param dpi:
        :return:
        '''

        # Pure Slip
        SVx = (self.PVX1 + self.PVX2 * dfz) * Fz * self.LVX * self.LMUX
        SHx = (self.PHX1 + self.PHX2 * dfz) * self.LHX
        kappa_x = kappa + SHx
        Kxk = 0 if kappa == 0 else Fz * (self.PKX1 + self.PKX2 * dfz) * np.exp(
            self.PKX3 * dfz) * self.LKX
        Ex = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz ** 2) \
            * (1 - self.PEX4 * np.sign(kappa_x)) * self.LEX
        Mewx = (self.PDX1 +
                self.PDX2 * dfz) * (1 - self.PDX3 * gamma**2) * self.LMUX
        Dx = Mewx * Fz
        Cx = self.PCX1 * self.LCX
        Bx = Kxk / (Cx * Dx)
        # Combined Slip
        Gxa = self.calc_combined_long_force_scalar(dfz, kappa,
                                                   alpha) if alpha else 1
        # Longitudinal Force (N)
        Bx_x_kappa_x = Bx * kappa_x
        Fx = (Dx *
              np.sin(Cx * np.arctan(Bx_x_kappa_x - Ex *
                                    (Bx_x_kappa_x - np.arctan(Bx_x_kappa_x))))
              + SVx) * Gxa

        return Fx, Kxk

    # TODO Rename this here and in `f_x`
    def calc_combined_long_force_scalar(self, dfz, kappa, alpha):
        SHxa = self.RHX1
        Exa = self.REX1 + self.REX2 * dfz
        Cxa = self.RCX1
        Bxa = self.RBX1 * np.cos(np.arctan(self.RBX2 * kappa)) * self.LXAL
        alpha_s = alpha + SHxa
        Bxa_x_alpha_s = Bxa * alpha_s
        Bxa_x_SHxa = Bxa * SHxa
        return (
            np.cos(Cxa * np.arctan(Bxa_x_alpha_s - Exa *
                                   (Bxa_x_alpha_s - np.arctan(Bxa_x_alpha_s))))) / \
               (np.cos(Cxa * np.arctan(Bxa_x_SHxa - Exa *
                                       (Bxa_x_SHxa - np.arctan(Bxa_x_SHxa)))))

    def m_z(self, alpha, kappa, gamma, Fz, Fy, Fx, dfz, dpi, By, Cy, SVy, SHy,
            Kya, Kxk):
        '''
        NOTE: Fy at 0 inclination_angle assumed approximately equal to calculated Fy to
        save computation time (for now)

        :param self:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param Fy:
        :param Fx:
        :param dfz:
        :param dpi:
        :param By:
        :param Cy:
        :param SVy:
        :param SHy:
        :param Kya:
        :param Kxk:
        :return:
        '''

        # Computational time not worth recalculating Fy with 0 inclination_angle given small inclination_angle angles
        Fyp0 = Fy
        gamma = gamma * self.LKZC

        # alpha_m = alpha when disregarding transient effects
        alpha_m = alpha
        SHt = self.QHZ1 + self.QHZ2 * dfz + (self.QHZ3 +
                                             self.QHZ4 * dfz) * gamma
        alpha_r = alpha_m + SHy + SVy / Kya
        alpha_t = alpha_m + SHt

        if not kappa or not alpha_m:  # if pure slip
            alpha_teq = alpha_t
            alpha_req = alpha_r
            s = 0
        else:
            alpha_teq = np.arctan(
                np.sqrt((np.tan(alpha_t))**2 +
                        (Kxk / Kya)**2 * kappa**2)) * np.sign(alpha_t)
            alpha_req = np.arctan(
                np.sqrt((np.tan(alpha_r))**2 +
                        (Kxk / Kya)**2 * kappa**2)) * np.sign(alpha_r)
            s = (self.SSZ1 + self.SSZ2 * (Fy / self.Fz0) + (self.SSZ3 + self.SSZ4 * dfz) * gamma) \
                * self.UNLOADED_RADIUS * self.LS

        # Pnneumatic Trail t
        Bt = (self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz ** 2) \
            * (1 + self.QBZ4 * gamma + self.QBZ5 * np.abs(gamma)) * self.LKY / self.LMUY
        Ct = self.QCZ1
        Dt = Fz * (self.QDZ1 + self.QDZ2 * dfz) * (1 + self.QDZ3 * np.abs(gamma) + self.QDZ4 * gamma ** 2) \
            * self.UNLOADED_RADIUS / self.Fz0 * self.LTR
        Et = (self.QEZ1 + self.QEZ2 * dfz + self.QEZ3 * dfz ** 2) \
            * (1 + (self.QEZ4 + self.QEZ5 * gamma) * (2 / np.pi) * np.arctan(Bt * Ct * alpha_t))
        Bt_x_alpha_teq = Bt * alpha_teq
        t = Dt * np.cos(Ct * np.arctan(Bt_x_alpha_teq - Et * (Bt_x_alpha_teq - np.arctan(Bt_x_alpha_teq)))) \
            * np.cos(alpha_m)

        # Residual Moment (Mzr)
        Dr = ((self.QDZ6 + self.QDZ7 * dfz) * self.LRES + (self.QDZ8 + self.QDZ9 * dfz) * gamma) \
            * Fz * self.UNLOADED_RADIUS * self.LMUY
        Br = self.QBZ9 * self.LKY / self.LMUY + self.QBZ10 * By * Cy
        Mzr = Dr * np.cos(np.arctan(Br * alpha_req)) * np.cos(alpha_m)

        return -t * Fyp0 + Mzr + s * Fx

    def m_x(self, gamma, Fz, Fy, dpi=0):
        '''

        :param self:
        :param gamma:
        :param Fz:
        :param Fy:
        :return:
        '''

        return (Fz * self.UNLOADED_RADIUS *
                (self.QSX1 - self.QSX2 * gamma + self.QSX3 * Fy / self.Fz0) *
                self.LMUX)


if __name__ == "__main__":

    # Choose and load tire model
    qt_helper = Dialogs(__file__ + 'get TIR')
    filename = str(qt_helper.select_file_dialog(accepted_file_types='*.TIR'))
    tire = Mf61(filepath=filename)

    # Create test ranges for slip angles and ratios
    slip_angles_input = np.linspace(np.deg2rad(-12), np.deg2rad(12), num=100)
    slip_ratios_input = np.linspace(-0.2, 0.2, num=100)
    inclinations = [
        np.deg2rad(-1.5),
        # np.deg2rad(-0.5), 0,
        # np.deg2rad(0.5),
        # np.deg2rad(1.5)
    ]

    # Create forces to plot
    # inclination_angle_rad = 1 * np.pi/180
    for inclination_angle_rad in inclinations:
        tire_forces = tire.create_tire_plot_test_data(slip_angles_input,
                                                      slip_ratios_input,
                                                      inclination_angle_rad,
                                                      1100, 55158)
        print(tire.tire_forces(0, 0, inclination_angle_rad, 1100, 55158))
        tire.plot_force(tire_forces)
