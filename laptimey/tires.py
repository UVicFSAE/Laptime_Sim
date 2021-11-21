"""
This file contains the tire model classes and their functions for the calculation of tire forces

- alpha = slip angle [deg]
- kappa = slip ratio [-]
- gamma = camber [deg]

Assumptions:
- Constant radius used
- Corrections for large camber angles not used for Mz

Created 2021, Contributors: Nigel Swab
"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame as df

from lib.pyqt_helper import Dialogs
# TODO: (Maybe) add tire radius calculations shown in MF61 resource material for more accuracy
# TODO: (Maybe) add tire stiffness calculations shown in MF61 resource material for more accuracy
# TODO: (Maybe) sort out why the heck the last term of Mx in Mf61 is so dang messed up

# TODO: (Maybe) Create an (abstract?) class that each MF class inherits from so there only needs to be one
#  import/force/create data method written. This, or just create a class that works for either. May also be worth
#  investigating saving tire files in another format with scaling coefficients or have scaling
#  coefficients separate for input or something '''


class Mf61:

    def __init__(self):
        pass

    __slots__ = ['FNOMIN', 'UNLOADED_RADIUS', 'LONGVL', 'NOMPRES', 'Fz0', 'LFZ0', 'LCX', 'LMUX',
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
                 'SSZ2', 'SSZ3', 'SSZ4', 'FITTYP', 'WIDTH', 'RIM_WIDTH', 'RIM_RADIUS', 'VERTICAL_STIFFNESS',
                 'PRESMIN', 'PRESMAX', 'KPUMIN', 'KPUMAX', 'ALPMIN', 'ALPMAX', 'CAMMIN', 'CAMMAX']

    # Model Information and Dimensions
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
    CAMMIN: float  # Minimum Valid Camber Angle (rad)
    CAMMAX: float  # Maximum Valid Camber Angle (rad)

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
    LYKA: float  # Scale factor of kappa influence on Fy
    LVYKA: float  # Scale factor of kappa induced Fy
    LS: float  # Scale factor of moment arm of Fx
    LKYC: float  # Scale factor of camber force stiffness
    LKZC: float  # Scale factor of camber torque stiffness
    LMU: float  # Scale factor with slip speed decaying friction
    LMX: float  # Scale factor of overturning couple
    LMY: float  # Scale factor of rolling resistance torque
    LVMX: float  # Overturning couple vertical shift

    # Longitudinal Coefficients
    PCX1: float  # Shape factor Cfx for longitudinal force
    PDX1: float  # Longitudinal friction Mux at Fznom
    PDX2: float  # Variation of friction Mux with load
    PDX3: float  # Variation of friction Mux with camber squared
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
    RBX3: float  # Influence of camber on stiffness for Fx combined
    RCX1: float  # Shape factor for combined slip Fx reduction
    REX1: float  # Curvature factor of combined Fx
    REX2: float  # Curvature factor of combined Fx with load
    RHX1: float  # Shift factor for combined slip Fx reduction

    # Overturning Coefficients
    SX1: float  # Vertical shift of overturning moment
    QSX2: float  # Camber induced overturning couple
    QSX3: float  # Fy induced overturning couple
    QSX4: float  # Mixed load lateral force and camber on Mx
    QSX5: float  # Load effect on Mx with lateral force and camber
    QSX6: float  # B-factor of load with Mx
    QSX7: float  # Camber with load on Mx
    QSX8: float  # Lateral force with load on Mx
    QSX9: float  # B-factor of lateral force with load on Mx
    QSX10: float  # Vertical force with camber on Mx
    QSX11: float  # B-factor of vertical force with camber on Mx
    QSX12: float  # Camber squared induced overturning moment
    QSX13: float  # Lateral force induced overturning moment
    QSX14: float  # Lateral force induced overturning moment with camber
    QPMX1: float  # Influence of inflation pressure on overturning moment

    # Lateral Coefficients
    PCY1: float  # Shape factor Cfy for lateral forces
    PDY1: float  # Lateral friction Muy
    PDY2: float  # Variation of friction Muy with load
    PDY3: float  # Variation of friction Muy with squared camber
    PEY1: float  # Lateral curvature Efy at Fznom
    PEY2: float  # Variation of curvature Efy with load
    PEY3: float  # Zero order camber dependency of curvature Efy
    PEY4: float  # Variation of curvature Efy with camber
    PEY5: float  # Variation of curvature Efy with camber squared
    PKY1: float  # Maximum value of stiffness Kfy/Fznom
    PKY2: float  # Load at which Kfy reaches maximum value
    PKY3: float  # Variation of Kfy/Fznom with camber
    PKY4: float  # Curvature of stiffness Kfy
    PKY5: float  # Peak stiffness variation with camber squared
    PKY6: float  # Fy camber stiffness factor
    PKY7: float  # Vertical load dependency of camber stiffness
    PHY1: float  # Horizontal shift Shy at Fznom
    PHY2: float  # Variation of shift Shy with load
    PVY1: float  # Vertical shift in Svy/Fz at Fznom
    PVY2: float  # Variation of shift Svy/Fz with load
    PVY3: float  # Variation of shift Svy/Fz with camber
    PVY4: float  # Variation of shift Svy/Fz with camber and load
    PPY1: float  # influence of inflation pressure on cornering stiffness
    PPY2: float  # influence of inflation pressure on dependency of nominal tyre load on cornering stiffness
    PPY3: float  # linear influence of inflation pressure on lateral peak friction
    PPY4: float  # quadratic influence of inflation pressure on lateral peak friction
    PPY5: float  # Influence of inflation pressure on camber stiffness
    RBY1: float  # Slope factor for combined Fy reduction
    RBY2: float  # Variation of slope Fy reduction with alpha
    RBY3: float  # Shift term for alpha in slope Fy reduction
    RBY4: float  # Influence of camber on stiffness of Fy combined
    RCY1: float  # Shape factor for combined Fy reduction
    REY1: float  # Curvature factor of combined Fy
    REY2: float  # Curvature factor of combined Fy with load
    RHY1: float  # Shift factor for combined Fy reduction
    RHY2: float  # Shift factor for combined Fy reduction with load
    RVY1: float  # Kappa induced side force Svyk/Muy*Fz at Fznom
    RVY2: float  # Variation of Svyk/Muy*Fz with load
    RVY3: float  # Variation of Svyk/Muy*Fz with camber
    RVY4: float  # Variation of Svyk/Muy*Fz with alpha
    RVY5: float  # Variation of Svyk/Muy*Fz with kappa
    RVY6: float  # Variation of Svyk/Muy*Fz with atan(kappa)

    # Aligning Moment Coefficients
    QBZ1: float  # Trail slope factor for trail Bpt at Fznom
    QBZ2: float  # Variation of slope Bpt with load
    QBZ3: float  # Variation of slope Bpt with load squared
    QBZ4: float  # Variation of slope Bpt with camber
    QBZ5: float  # Variation of slope Bpt with absolute camber
    QBZ6: float  # Camber influence Bt
    QBZ9: float  # Factor for scaling factors of slope factor Br of Mzr
    QBZ10: float  # Factor for dimensionless cornering stiffness of Br of Mzr
    QCZ1: float  # Shape factor Cpt for pneumatic trail
    QDZ1: float  # Peak trail Dpt = Dpt*(Fz/Fznom*R0)
    QDZ2: float  # Variation of peak Dpt" with load
    QDZ3: float  # Variation of peak Dpt" with camber
    QDZ4: float  # Variation of peak Dpt" with camber squared
    QDZ6: float  # Peak residual torque Dmr" = Dmr/(Fz*R0)
    QDZ7: float  # Variation of peak factor Dmr" with load
    QDZ8: float  # Variation of peak factor Dmr" with camber
    QDZ9: float  # Variation of peak factor Dmr" with camber and load
    QDZ10: float  # Variation of peak factor Dmr with camber squared
    QDZ11: float  # Variation of Dmr with camber squared and load
    QEZ1: float  # Trail curvature Ept at Fznom
    QEZ2: float  # Variation of curvature Ept with load
    QEZ3: float  # Variation of curvature Ept with load squared
    QEZ4: float  # Variation of curvature Ept with sign of Alpha-t
    QEZ5: float  # Variation of Ept with camber and sign Alpha-t
    QHZ1: float  # Trail horizontal shift Sht at Fznom
    QHZ2: float  # Variation of shift Sht with load
    QHZ3: float  # Variation of shift Sht with camber
    QHZ4: float  # Variation of shift Sht with camber and load
    PPZ1: float  # effect of inflation pressure on length of pneumatic trail
    PPZ2: float  # Influence of inflation pressure on residual aligning torque
    SSZ1: float  # Nominal value of s/R0: effect of Fx on Mz
    SSZ2: float  # Variation of distance s/R0 with Fy/Fznom
    SSZ3: float  # Variation of distance s/R0 with camber
    SSZ4: float  # Variation of distance s/R0 with load and camber

    @classmethod
    def load_model_from_tir(cls, filepath: str) -> None:
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
                if key in cls.__slots__:
                    value = value.split(sep='$', maxsplit=1)
                    setattr(cls, key, float(value[0]))
                else:
                    pass
        cls.Fz0 = cls.FNOMIN * cls.LFZ0

        if cls.FITTYP != 61:
            raise ValueError(f'Expecting Mf61, got Mf{cls.FITTYP}')
        else:
            print(f'Tire model loaded \n')

    @classmethod
    def tire_forces(cls, alpha, kappa, gamma, Fz, tire_pressure):
        '''

        :param cls:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param tire_pressure:
        :return:
        '''

        # Initializations
        dfz = (Fz - cls.Fz0) / cls.Fz0
        dpi = (tire_pressure - cls.NOMPRES) / cls.NOMPRES

        Fy, By, Cy, SVy, SHy, Kya = cls.f_y(alpha, kappa, gamma, Fz, dfz, dpi)
        Fx, Kxk = cls.f_x(alpha, kappa, gamma, Fz, dfz, dpi)
        Mz = cls.m_z(alpha, kappa, gamma, Fz, Fy, Fx, dfz, dpi, By, Cy, SVy, SHy, Kya, Kxk)
        Mx = cls.m_x(gamma, Fz, Fy, dpi)

        return Fy, Fx, Mz, Mx

    @classmethod
    def f_y(cls, alpha, kappa, gamma, Fz, dfz, dpi):
        '''Calculates and returns the lateral force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        camber (gamma), normal force  (Fz), and tire pressure

        :param cls:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param dfz:
        :param dpi:
        :return:
        '''

        # Pure slip
        SVyg = Fz * (cls.PVY3 + cls.PVY4 * dfz) * gamma * cls.LKYC * cls.LMUY
        SVy0 = Fz * (cls.PVY1 + cls.PVY2 * dfz) * cls.LVY * cls.LMUY
        SVy = SVy0 + SVyg
        Kyg = (cls.PKY6 + cls.PKY7 * dfz) * (1 + cls.PPY5 * dpi) * Fz * cls.LKYC
        Kya = (1 - cls.PKY3 * abs(gamma)) * cls.PKY1 * cls.Fz0 * (1 + cls.PPY1 * dpi) \
              * np.sin(cls.PKY4 * np.arctan(Fz / ((cls.PKY2 + cls.PKY5 * np.power(gamma, 2))
                                                     * (1 + cls.PPY2 * dpi) * cls.Fz0))) * cls.LKY

        SHy0 = (cls.PHY1 + cls.PHY2 * dfz) * cls.LHY
        SHyg = (Kyg * gamma - SVyg) / Kya
        SHy = SHy0 + SHyg
        alpha_y = alpha + SHy
        Ey = (cls.PEY1 + cls.PEY2 * dfz) * \
             (1 + cls.PEY5 * gamma ** 2 - (cls.PEY3 + cls.PEY4 * gamma) * np.sign(alpha_y)) * cls.LEY
        Mewy = (cls.PDY1 + cls.PDY2 * dfz) * (1 - cls.PDY3 * gamma ** 2) * \
               (1 + cls.PPY3 * dpi + cls.PPY4 * gamma ** 2) * cls.LMUY
        Dy = Mewy * Fz
        Cy = cls.PCY1 * cls.LCY
        By = Kya / (Cy * Dy)
        SVyk = 0
        Gyk = 1

        # Combined Slip (if there is a slip ratio)
        if kappa:
            SHyk = cls.RHY1 + cls.RHY2 * dfz
            Eyk = cls.REY1 + cls.REY2 * dfz
            Cyk = cls.RCY1
            Byk = (cls.RBY1 + cls.RBY4 * gamma ** 2) * \
                  np.cos(np.arctan(cls.RBY2 * (alpha - cls.RBY3))) * cls.LYKA
            kappa_s = kappa + SHyk
            Gyk = (np.cos(Cyk * np.arctan(Byk * kappa_s - Eyk * (Byk * kappa_s - np.arctan(Byk * kappa_s))))) / \
                  (np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk)))))
            DVyk = Mewy * Fz * (cls.RVY1 + cls.RVY2 * dfz + cls.RVY3 * gamma) * \
                   np.cos(np.arctan(cls.RVY4 * alpha))
            SVyk = DVyk * np.sin(cls.RVY5 * np.arctan(cls.RVY6)) * cls.LVYKA

        # Pure Lateral Force
        By_x_alpha_y = By * alpha_y
        Fyp = Dy * np.sin(Cy * np.arctan(By_x_alpha_y - Ey * (By_x_alpha_y - np.arctan(By_x_alpha_y)))) + SVy

        # Combined Lateral Force ( Fyp == Fy if kappa == 0)
        Fy = Gyk * Fyp + SVyk
        return Fy, By, Cy, SVy, SHy, Kya

    @classmethod
    def f_x(cls, alpha, kappa, gamma, Fz, dfz, dpi):
        ''' Calculates and returns the longitudinal force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        camber (gamma), normal force  (Fz), and tire pressure

        :param cls:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param dfz:
        :param dpi:
        :return:
        '''

        # Pure Slip
        SVx = (cls.PVX1 + cls.PVX2 * dfz) * Fz * cls.LVX * cls.LMUX
        SHx = (cls.PHX1 + cls.PHX2 * dfz) * cls.LHX
        kappa_x = kappa + SHx
        Kxk = (cls.PKX1 + cls.PKX2 * dfz) * np.exp(cls.PKX3 * dfz) * \
              (1 + cls.PPX1 * dpi + cls.PPX2 * dpi ** 2) * Fz * cls.LKX
        Ex = (cls.PEX1 + cls.PEX2 * dfz + cls.PEX3 * dfz ** 2) * \
             (1 - cls.PEX4 * np.sign(kappa_x)) * cls.LEX
        Mewx = (cls.PDX1 + cls.PDX2 * dfz) * (1 - cls.PDX3 * gamma ** 2) * \
               (1 + cls.PPX3 * dpi + cls.PPX4 * dpi ** 2) * cls.LMUX
        Dx = Mewx * Fz
        Cx = cls.PCX1 * cls.LCX
        Bx = Kxk / (Cx * Dx)
        Gxa = 1

        # Combined Slip
        if alpha:
            SHxa = cls.RHX1
            Exa = cls.REX1 + cls.REX2 * dfz
            Cxa = cls.RCX1
            Bxa = (cls.RBX1 + cls.RBX3 * gamma ** 2) * np.cos(np.arctan(cls.RBX2 * kappa)) * cls.LXAL
            alpha_s = alpha + SHxa
            Bxa_x_alpha_s = Bxa * alpha_s
            Bxa_x_SHxa = Bxa * SHxa
            Gxa = (np.cos(Cxa * np.arctan(Bxa_x_alpha_s - Exa * (Bxa_x_alpha_s - np.arctan(Bxa_x_alpha_s))))) / \
                  (np.cos(Cxa * np.arctan(Bxa_x_SHxa - Exa * (Bxa_x_SHxa - np.arctan(Bxa_x_SHxa)))))

        # Longitudinal Force (N)
        Bx_kappa_x = Bx * kappa_x
        Fx = (Dx * np.sin(Cx * np.arctan(Bx_kappa_x - Ex * (Bx_kappa_x - np.arctan(Bx_kappa_x)))) + SVx) * Gxa
        return Fx, Kxk

    @classmethod
    def m_z(cls, alpha, kappa, gamma, Fz, Fy, Fx, dfz, dpi, By, Cy, SVy, SHy, Kya, Kxk):
        '''
        NOTE: Fy at 0 camber assumed approximately equal to calculated Fy to save computation time (for now)

        :param cls:
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

        # Computational time not worth recalculating Fy with 0 camber given small camber angles
        Fyp0 = Fy
        # TODO: Test to see how much of a difference the 0 camber Fy makes
        # Fyp0, By, Cy, SVy, SHy, Kya = cls.f_y(alpha, kappa, gamma, Fz, dfz, dpi)

        # alpha_m = alpha when disregarding transient effects
        alpha_m = alpha
        SHt = cls.QHZ1 + cls.QHZ2 * dfz + (cls.QHZ3 + cls.QHZ4 * dfz) * gamma
        alpha_r = alpha_m + SHy + SVy / Kya
        alpha_t = alpha_m + SHt

        if not kappa or not alpha:  # if pure slip
            alpha_teq = alpha_t
            alpha_req = alpha_r
            s = 0
        else:
            alpha_teq = np.arctan(np.sqrt((np.tan(alpha_t)) ** 2 + (Kxk / Kya) ** 2 * kappa ** 2)) * np.sign(alpha_t)
            alpha_req = np.arctan(np.sqrt((np.tan(alpha_r)) ** 2 + (Kxk / Kya) ** 2 * kappa ** 2)) * np.sign(alpha_r)
            s = (cls.SSZ1 + cls.SSZ2 * (Fy / cls.Fz0) + (cls.SSZ3 + cls.SSZ4 * dfz) * gamma) \
                * cls.UNLOADED_RADIUS * cls.LS

        # Pnneumatic Trail t
        Bt = (cls.QBZ1 + cls.QBZ2 * dfz + cls.QBZ3 * dfz ** 2) \
            * (1 + cls.QBZ4 * gamma + cls.QBZ5 * np.abs(gamma) + cls.QBZ6 * gamma ** 2) * cls.LKY / cls.LMUY
        # Note: Have had odd QBZ4 Values from Optimum tire before, resulting in wonky Mz
        Ct = cls.QCZ1
        Dt = Fz * (cls.QDZ1 + cls.QDZ2 * dfz) * (1 - cls.PPZ1 * dpi) \
            * (1 + cls.QDZ3 * gamma + cls.QDZ4 * gamma ** 2) * cls.UNLOADED_RADIUS / cls.Fz0 * cls.LTR
        Et = (cls.QEZ1 + cls.QEZ2 * dfz + cls.QEZ3 * dfz ** 2) \
            * (1 + (cls.QEZ4 + cls.QEZ5 * gamma) * (2 / np.pi) * np.arctan(Bt * Ct * alpha_t))
        Bt_x_alpha_teq = Bt * alpha_teq
        t = Dt * np.cos(Ct * np.arctan(Bt_x_alpha_teq - Et * (Bt_x_alpha_teq - np.arctan(Bt_x_alpha_teq)))) \
            * np.cos(alpha_m)

        # Residual Moment (Mzr)
        Dr = ((cls.QDZ6 + cls.QDZ7 * dfz) * cls.LRES + (cls.QDZ8 + cls.QDZ9 * dfz) *
              (1 - cls.PPZ2 * dpi) * gamma * cls.LKZC + (cls.QDZ10 + cls.QDZ11 * dfz) *
              gamma * np.abs(gamma) * cls.LKZC) * Fz * cls.UNLOADED_RADIUS * cls.LMUY
        Br = cls.QBZ9 * cls.LKY / cls.LMUY + cls.QBZ10 * By * Cy
        Mzr = Dr * np.cos(np.arctan(Br * alpha_req)) * np.cos(alpha_m)

        # Aligning Moment
        Mz = -t * Fyp0 + Mzr + s * Fx

        return Mz

    @classmethod
    def m_x(cls, gamma, Fz, Fy, dpi):
        '''

        :param cls:
        :param gamma:
        :param Fz:
        :param Fy:
        :param dpi:
        :return:
        '''

        #  Fy normalized by Fz0
        Fy_div_Fz0 = Fy / cls.Fz0

        Mx = cls.UNLOADED_RADIUS * Fz * cls.LMX \
            * (cls.QSX1 * cls.LVMX
               - cls.QSX2 * gamma * (1 + cls.QPMX1 * dpi)
               - cls.QSX12 * gamma * np.abs(gamma)
               + cls.QSX3 * Fy_div_Fz0
               + cls.QSX4 * np.cos(cls.QSX5 * np.arctan((cls.QSX6 * Fz / cls.Fz0) ** 2))
               * np.sin(cls.QSX7 * gamma + cls.QSX8 * np.arctan(cls.QSX9 * Fy_div_Fz0))
               + cls.QSX10 * np.arctan(cls.QSX11 * Fy_div_Fz0) * gamma) \
            # + cls.UNLOADED_RADIUS * Fy * cls.LMX * (cls.QSX13 + cls.QSX14 * np.abs(gamma))
        'Not sure why the last term throws everything off, OptimumTire may not fit properly for it?'

        return Mx

    @classmethod
    def create_tire_plot_test_data(cls, alpha_range, kappa_range, gamma, Fz, tire_pressure):

        # Initialize lists for forces
        fy_forces = []
        fx_forces = []
        mz_forces = []
        mx_forces = []

        kappa = 0
        for alpha in alpha_range:
            Fy, _, Mz, Mx = cls.tire_forces(alpha, kappa, gamma, Fz, tire_pressure)
            fy_forces.append(Fy)
            mz_forces.append(Mz)
            mx_forces.append(Mx)

        alpha = 0
        for kappa in kappa_range:
            _, Fx, _, _ = cls.tire_forces(alpha, kappa, gamma, Fz, tire_pressure)
            fx_forces.append(Fx)
        alpha_range = np.rad2deg(alpha_range)
        headings = ['Slip Angle [deg]', 'Slip Ratio [-]', 'Fy [N]', 'Fx [N]', 'Mz [Nm]', 'Mx [Nm]']
        forces = df(data=list(zip(alpha_range, kappa_range, fy_forces, fx_forces, mz_forces, mx_forces)),
                    columns=headings)
        return forces


class Mf52:

    """Equations from https://drive.google.com/file/d/1qjyM6F8YzKPEFXYUvE8Ptty8vhaR5SVw/view?usp=sharing"""

    __slots__ = ['FNOMIN', 'UNLOADED_RADIUS', 'LONGVL', 'Fz0', 'LFZ0', 'LCX', 'LMUX', 'LEX',
                 'LKX', 'LHX', 'LVX', 'LCY', 'LMUY', 'LEY', 'LKY', 'LHY', 'LVY', 'LTR', 'LRES',
                 'LXAL', 'LYKA', 'LVYKA', 'LS', 'LKYC', 'LKZC', 'LMUV', 'LMX', 'LMY', 'PCX1',
                 'PDX1', 'PDX2', 'PDX3', 'PEX1', 'PEX2', 'PEX3', 'PEX4', 'PKX1', 'PKX2', 'PKX3',
                 'PHX1', 'PHX2', 'PVX1', 'PVX2', 'RBX1', 'RBX2', 'RCX1', 'REX1', 'REX2', 'PTX1',
                 'PTX2', 'PTX3', 'RHX1', 'QSX1', 'QSX2', 'QSX3', 'PCY1', 'PDY1', 'PDY2', 'PDY3',
                 'PEY1', 'PEY2', 'PEY3', 'PEY4', 'PKY1', 'PKY2', 'PKY3', 'PHY1', 'PHY2', 'PHY3',
                 'PVY1', 'PVY2', 'PVY3', 'PVY4', 'RBY1','RBY2', 'RBY3', 'RCY1', 'REY1', 'REY2',
                 'RHY1', 'RHY2', 'RVY1', 'RVY2', 'RVY3', 'RVY4', 'RVY5', 'RVY6', 'PTY1', 'PTY2',
                 'QBZ1', 'QBZ2', 'QBZ3', 'QBZ4','QBZ5', 'QBZ9', 'QBZ10', 'QCZ1', 'QDZ1', 'QDZ2',
                 'QDZ3', 'QDZ4', 'QDZ6', 'QDZ7', 'QDZ8', 'QDZ9', 'QEZ1', 'QEZ2', 'QEZ3', 'QEZ4',
                 'QEZ5', 'QHZ1', 'QHZ2', 'QHZ3', 'QHZ4', 'SSZ1','SSZ2', 'SSZ3', 'SSZ4', 'QTZ1',
                 'MBELT', 'FITTYP', 'WIDTH', 'RIM_WIDTH', 'RIM_RADIUS', 'VERTICAL_STIFFNESS',
                 'PRESMIN', 'PRESMAX', 'KPUMIN', 'KPUMAX', 'ALPMIN', 'ALPMAX', 'CAMMIN', 'CAMMAX']

    # Model Information and Dimensions
    FITTYP: float  # Magic formula number (ie. 52 = MF5.2)
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
    CAMMIN: float  # Minimum Valid Camber Angle (rad)
    CAMMAX: float  # Maximum Valid Camber Angle (rad)

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
    LKYC: float  # Scale factor of camber force stiffness
    LKZC: float  # Scale factor of camber torque stiffness
    LMUV: float  # Scale factor with slip speed decaying friction
    LMX: float  # Scale factor of overturning couple
    LMY: float  # Scale factor of rolling resistance torque

    # Longitudinal Coefficients
    PCX1: float  # Shape factor Cfx for longitudinal force
    PDX1: float  # Longitudinal friction Mux at Fznom
    PDX2: float  # Variation of friction Mux with load
    PDX3: float  # Variation of friction Mux with camber square
    PEX1: float  # Longitudinal curvature Efx at Fznom
    PEX2: float  # Variation of curvature Efx with load
    PEX3: float  # Variation of curvature Efx with load square
    PEX4: float  # Factor in curvature Efx while driving
    PKX1: float  # Longitudinal slip stiffness Kfx/Fz at Fznom
    PKX2: float  # Variation of slip stiffness Kfx/Fz with
    PKX3: float  # Exponent in slip stiffness Kfx/Fz with loa
    PHX1: float  # Horizontal shift Shx at Fznom
    PHX2: float  # Variation of shift Shx with load
    PVX1: float  # Vertical shift Svx/Fz at Fznom
    PVX2: float  # Variation of shift Svx/Fz with load
    RBX1: float  # Slope factor for combined slip Fx reduction
    RBX2: float  # Variation of slope Fx reduction with kappa
    RCX1: float  # Shape factor for combined slip Fx reduction
    REX1: float  # Curvature factor of combined Fx
    REX2: float  # Curvature factor of combined Fx with load
    RHX1: float  # Shift factor for combined slip Fx reducti
    PTX1: float  # Relaxation length SigKap0/Fz at Fznom
    PTX2: float  # Variation of SigKap0/Fz with load
    PTX3: float  # Variation of SigKap0/Fz with exponent

    # Overturning Coefficients
    QSX1: float  # Lateral force induced overturning moment
    QSX2: float  # Camber induced overturning couple
    QSX3: float  # Fy induced overturning couple

    # Lateral Coefficients
    PCY1: float  # Shape factor Cfy for lateral forces
    PDY1: float  # Lateral friction Muy
    PDY2: float  # Variation of friction Muy with load
    PDY3: float  # Variation of friction Muy with squared camber
    PEY1: float  # Lateral curvature Efy at Fznom
    PEY2: float  # Variation of curvature Efy with load
    PEY3: float  # Zero order camber dependency of curvature Efy
    PEY4: float  # Variation of curvature Efy with camber
    PKY1: float  # Maximum value of stiffness Kfy/Fznom
    PKY2: float  # Load at which Kfy reaches maximum value
    PKY3: float  # Variation of Kfy/Fznom with camber
    PHY1: float  # Horizontal shift Shy at Fznom
    PHY2: float  # Variation of shift Shy with load
    PHY3: float  # Variation of shift Shy with camber
    PVY1: float  # Vertical shift in Svy/Fz at Fznom
    PVY2: float  # Variation of shift Svy/Fz with load
    PVY3: float  # Variation of shift Svy/Fz with camber
    PVY4: float  # Variation of shift Svy/Fz with camber and load
    RBY1: float  # Slope factor for combined Fy reduction
    RBY2: float  # Variation of slope Fy reduction with alpha
    RBY3: float  # Shift term for alpha in slope Fy reduction
    RCY1: float  # Shape factor for combined Fy reduction
    REY1: float  # Curvature factor of combined Fy
    REY2: float  # Curvature factor of combined Fy with load
    RHY1: float  # Shift factor for combined Fy reduction
    RHY2: float  # Shift factor for combined Fy reduction with loa
    RVY1: float  # Kappa induced side force Svyk/Muy*Fz at Fznom
    RVY2: float  # Variation of Svyk/Muy*Fz with load
    RVY3: float  # Variation of Svyk/Muy*Fz with camber
    RVY4: float  # Variation of Svyk/Muy*Fz with alpha
    RVY5: float  # Variation of Svyk/Muy*Fz with kappa
    RVY6: float  # Variation of Svyk/Muy*Fz with atan(kappa)
    PTY1: float  # Peak value of relaxation length SigAlp0/R0
    PTY2: float  # Value of Fz/Fznom where SigAlp0 is extreme

    # Aligning Moment Coefficients
    QBZ1: float  # Trail slope factor for trail Bpt at Fznom
    QBZ2: float  # Variation of slope Bpt with load
    QBZ3: float  # Variation of slope Bpt with load squared
    QBZ5: float  # Variation of slope Bpt with absolute camber
    QBZ4: float  # Variation of slope Bpt with camber
    QBZ9: float  # Slope factor Br of residual torque Mzr
    QBZ10: float  # Slope factor Br of residual torque Mzr
    QCZ1: float  # Shape factor Cpt for pneumatic trail
    QDZ1: float  # Peak trail Dpt" = Dpt*(Fz/Fznom*R0)
    QDZ2: float  # Variation of peak Dpt" with load
    QDZ3: float  # Variation of peak Dpt" with camber
    QDZ4: float  # Variation of peak Dpt" with camber squared
    QDZ6: float  # Peak residual torque Dmr" = Dmr/(Fz*R0)
    QDZ7: float  # Variation of peak factor Dmr" with load
    QDZ8: float  # $Variation of peak factor Dmr" with camber
    QDZ9: float  # Variation of peak factor Dmr" with camber and load
    QEZ1: float  # Trail curvature Ept at Fznom
    QEZ2: float  # Variation of curvature Ept with load
    QEZ3: float  # $Variation of curvature Ept with load squared
    QEZ4: float  # $Variation of curvature Ept with sign of Alpha-t
    QEZ5: float  # Variation of Ept with camber and sign Alpha-t
    QHZ1: float  # Trail horizontal shift Sht at Fznom
    QHZ2: float  # Variation of shift Sht with load
    QHZ3: float  # Variation of shift Sht with camber
    QHZ4: float  # Variation of shift Sht with camber and load
    SSZ1: float  # Nominal value of s/R0: effect of Fx on Mz
    SSZ2: float  # Variation of distance s/R0 with Fy/Fznom
    SSZ3: float  # Variation of distance s/R0 with camber
    SSZ4: float  # Variation of distance s/R0 with load and camber
    QTZ1: float  # Gyration torque constant
    MBELT: float  # Belt mass of the wheel

    @classmethod
    def load_model_from_tir(cls, filepath: str) -> None:
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
                if key in cls.__slots__:
                    value = value.split(sep='$', maxsplit=1)
                    setattr(cls, key, float(value[0]))
                else:
                    pass
        cls.Fz0 = cls.FNOMIN * cls.LFZ0

        if cls.FITTYP != 52:
            raise ValueError(f'Expecting Mf52, got Mf{cls.FITTYP}')
        else:
            print(f'Tire model loaded \n')

    @classmethod
    def tire_forces(cls, alpha, kappa, gamma, Fz, tire_pressure):
        '''

        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param tire_pressure:
        :return:
        '''

        # Initializations
        dfz = (Fz - cls.Fz0) / cls.Fz0

        Fy, By, Cy, SVy, SHy, Ky = cls.f_y(alpha, kappa, gamma, Fz, dfz)
        Fx, Kx = cls.f_x(alpha, kappa, gamma, Fz, dfz)
        Mz = cls.m_z(alpha, kappa, gamma, Fz, Fy, Fx, dfz, By, Cy, SVy, SHy, Ky, Kx)
        Mx = cls.m_x(gamma, Fz, Fy)

        return Fy, Fx, Mz, Mx

    @classmethod
    def f_y(cls, alpha, kappa, gamma, Fz, dfz):
        '''Calculates and returns the lateral force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        camber (gamma), normal force  (Fz), and tire pressure

        :param cls:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param dfz:
        :return:
        '''

        # Pure slip
        SVy0 = Fz * (cls.PVY1 + cls.PVY2 * dfz) * cls.LVY * cls.LMUY
        SVyg = Fz * (cls.PVY3 + cls.PVY4 * dfz) * gamma * cls.LKYC * cls.LMUY
        SVy = SVy0 + SVyg
        SHy = (cls.PHY1 + cls.PHY2 * dfz) * cls.LHY + cls.PHY3 * gamma
        alpha_y = alpha + SHy
        Ky = cls.PKY1 * cls.Fz0 * np.sin(np.arctan(Fz / (cls.PKY2 * cls.Fz0))) \
            * (1 - cls.PKY3 * np.abs(gamma)) * cls.LKY
        Ey = (cls.PEY1 + cls.PEY2 * dfz) * (1 - (cls.PEY3 + cls.PEY4 * gamma) * np.sign(alpha_y)) * cls.LEY
        Mewy = (cls.PDY1 + cls.PDY2 * dfz) * (1 - cls.PDY3 * gamma ** 2) * cls.LMUY
        Dy = Mewy * Fz
        Cy = cls.PCY1 * cls.LCY
        By = Ky / (Cy * Dy)
        SVyk = 0
        Gyk = 1

        # Combined Slip (if there is a slip ratio)
        if kappa:
            SHyk = cls.RHY1 + cls.RHY2 * dfz
            Eyk = cls.REY1 + cls.REY2 * dfz
            Cyk = cls.RCY1
            Byk = cls.RBY1 * np.cos(np.arctan(cls.RBY2 * (alpha - cls.RBY3))) * cls.LYKA
            kappa_s = kappa + SHyk
            Gyk = (np.cos(Cyk * np.arctan(Byk * kappa_s - Eyk * (Byk * kappa_s - np.arctan(Byk * kappa_s))))) / \
                  (np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk)))))
            DVyk = Mewy * Fz * (cls.RVY1 + cls.RVY2 * dfz + cls.RVY3 * gamma) * \
                   np.cos(np.arctan(cls.RVY4 * alpha))
            SVyk = DVyk * np.sin(cls.RVY5 * np.arctan(cls.RVY6)) * cls.LVYKA

        # Pure Lateral Force
        Fyp = Dy * np.sin(Cy * np.arctan(By * alpha_y - Ey * (By * alpha_y - np.arctan(By * alpha_y)))) + SVy

        # Combined Lateral Force ( Fyp == Fy if kappa == 0)
        Fy = Gyk * Fyp + SVyk

        return Fy, By, Cy, SVy, SHy, Ky

    @classmethod
    def f_x(cls, alpha: float, kappa: float, gamma: float, Fz: float, dfz: float):
        ''' Calculates and returns the longitudinal force produced by a tire given a slip angle (alpha), slip ratio (kappa),
        camber (gamma), normal force  (Fz), and tire pressure

        :param cls:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param dfz:
        :return:
        '''

        # Pure Slip
        SVx = (cls.PVX1 + cls.PVX2 * dfz) * Fz * cls.LVX * cls.LMUX
        SHx = (cls.PHX1 + cls.PHX2 * dfz) * cls.LHX
        kappa_x = kappa + SHx
        Kx = 0 if kappa == 0 else Fz * (cls.PKX1 + cls.PKX2 * dfz) * np.exp(cls.PKX3 * dfz) * cls.LKX
        Ex = (cls.PEX1 + cls.PEX2 * dfz + cls.PEX3 * dfz ** 2) \
            * (1 - cls.PEX4 * np.sign(kappa_x)) * cls.LEX
        Mewx = (cls.PDX1 + cls.PDX2 * dfz) * (1 - cls.PDX3 * gamma ** 2) * cls.LMUX
        Dx = Mewx * Fz
        Cx = cls.PCX1 * cls.LCX
        Bx = Kx / (Cx * Dx)
        Gxa = 1

        # Combined Slip
        if alpha:
            SHxa = cls.RHX1
            Exa = cls.REX1 + cls.REX2 * dfz
            Cxa = cls.RCX1
            Bxa = cls.RBX1 * np.cos(np.arctan(cls.RBX2 * kappa)) * cls.LXAL
            alpha_s = alpha + SHxa
            Bxa_x_alpha_s = Bxa * alpha_s
            Bxa_x_SHxa = Bxa * SHxa
            Gxa = (np.cos(Cxa * np.arctan(Bxa_x_alpha_s - Exa * (Bxa_x_alpha_s - np.arctan(Bxa_x_alpha_s))))) / \
                  (np.cos(Cxa * np.arctan(Bxa_x_SHxa - Exa * (Bxa_x_SHxa - np.arctan(Bxa_x_SHxa)))))

        # Longitudinal Force (N)
        Bx_x_kappa_x = Bx * kappa_x
        Fx = (Dx * np.sin(Cx * np.arctan(Bx_x_kappa_x - Ex * (Bx_x_kappa_x - np.arctan(Bx_x_kappa_x)))) + SVx) * Gxa

        return Fx, Kx

    @classmethod
    def m_z(cls, alpha, kappa, gamma, Fz, Fy, Fx, dfz, By, Cy, SVy, SHy, Ky, Kx):
        '''
        NOTE: Fy at 0 camber assumed approximately equal to calculated Fy to save computation time (for now)

        :param cls:
        :param alpha:
        :param kappa:
        :param gamma:
        :param Fz:
        :param Fy:
        :param Fx:
        :param dfz:
        :param By:
        :param Cy:
        :param SVy:
        :param SHy:
        :param Ky:
        :param Kx:
        :return:
        '''

        # Computational time not worth recalculating Fy with 0 camber given small camber angles
        Fyp0 = Fy
        gamma = gamma * cls.LKZC

        # alpha_m = alpha when disregarding transient effects
        alpha_m = alpha
        SHt = cls.QHZ1 + cls.QHZ2 * dfz + (cls.QHZ3 + cls.QHZ4 * dfz) * gamma
        alpha_r = alpha_m + SHy + SVy / Ky
        alpha_t = alpha_m + SHt

        if not kappa or not alpha:  # if pure slip
            alpha_teq = alpha_t
            alpha_req = alpha_r
            s = 0
        else:
            alpha_teq = np.arctan(np.sqrt((np.tan(alpha_t)) ** 2 + (Kx / Ky) ** 2 * kappa ** 2)) * np.sign(alpha_t)
            alpha_req = np.arctan(np.sqrt((np.tan(alpha_r)) ** 2 + (Kx / Ky) ** 2 * kappa ** 2)) * np.sign(alpha_r)
            s = (cls.SSZ1 + cls.SSZ2 * (Fy / cls.Fz0) + (cls.SSZ3 + cls.SSZ4 * dfz) * gamma) \
                * cls.UNLOADED_RADIUS * cls.LS

        # Pnneumatic Trail t
        Bt = (cls.QBZ1 + cls.QBZ2 * dfz + cls.QBZ3 * dfz ** 2) \
            * (1 + cls.QBZ4 * gamma + cls.QBZ5 * np.abs(gamma)) * cls.LKY / cls.LMUY
        Ct = cls.QCZ1
        Dt = Fz * (cls.QDZ1 + cls.QDZ2 * dfz) * (1 + cls.QDZ3 * np.abs(gamma) + cls.QDZ4 * gamma ** 2) \
            * cls.UNLOADED_RADIUS / cls.Fz0 * cls.LTR
        Et = (cls.QEZ1 + cls.QEZ2 * dfz + cls.QEZ3 * dfz ** 2) \
            * (1 + (cls.QEZ4 + cls.QEZ5 * gamma) * (2 / np.pi) * np.arctan(Bt * Ct * alpha_t))
        Bt_x_alpha_teq = Bt * alpha_teq
        t = Dt * np.cos(Ct * np.arctan(Bt_x_alpha_teq - Et * (Bt_x_alpha_teq - np.arctan(Bt_x_alpha_teq)))) \
            * np.cos(alpha_m)

        # Residual Moment (Mzr)
        Dr = ((cls.QDZ6 + cls.QDZ7 * dfz) * cls.LRES + (cls.QDZ8 + cls.QDZ9 * dfz) * gamma) \
            * Fz * cls.UNLOADED_RADIUS * cls.LMUY
        Br = cls.QBZ9 * cls.LKY / cls.LMUY + cls.QBZ10 * By * Cy
        Mzr = Dr * np.cos(np.arctan(Br * alpha_req)) * np.cos(alpha_m)

        # Aligning Moment
        Mz = -t * Fyp0 + Mzr + s * Fx

        return Mz

    @classmethod
    def m_x(cls, gamma, Fz, Fy):
        '''

        :param cls:
        :param gamma:
        :param Fz:
        :param Fy:
        :return:
        '''

        Mx = Fz * cls.UNLOADED_RADIUS * (cls.QSX1 - cls.QSX2 * gamma + cls.QSX3 * Fy / cls.Fz0) * cls.LMUX

        return Mx

    @classmethod
    def create_tire_plot_test_data(cls, alpha_range, kappa_range, gamma, Fz, tire_pressure):
        # Initialize lists for forces
        fy_forces = []
        fx_forces = []
        mz_forces = []
        mx_forces = []

        kappa = 0
        for alpha in alpha_range:
            Fy, _, Mz, Mx = cls.tire_forces(alpha, kappa, gamma, Fz, tire_pressure)
            fy_forces.append(Fy)
            mz_forces.append(Mz)
            mx_forces.append(Mx)

        alpha = 0
        for kappa in kappa_range:
            _, Fx, _, _ = cls.tire_forces(alpha, kappa, gamma, Fz, tire_pressure)
            fx_forces.append(Fx)
        alpha_range = np.rad2deg(alpha_range)
        headings = ['Slip Angle [deg]', 'Slip Ratio [-]', 'Fy [N]', 'Fx [N]', 'Mz [Nm]', 'Mx [Nm]']
        forces = df(data=list(zip(alpha_range, kappa_range, fy_forces, fx_forces, mz_forces, mx_forces)),
                    columns=headings)
        return forces


def plot_force(data):
    # Fy
    x = data['Slip Angle [deg]']
    y = data['Fy [N]']
    plt.figure('Fy')
    plt.plot(x, y)

    # Fx
    x = data['Slip Ratio [-]']
    y = data['Fx [N]']
    plt.figure('Fx')
    plt.plot(x, y)

    # Mz
    x = data['Slip Angle [deg]']
    y = data['Mz [Nm]']
    plt.figure('Mz')
    plt.plot(x, y)

    # Mx
    y = data['Mx [Nm]']
    plt.figure('Mx')
    plt.plot(x, y)

    plt.show()


if __name__ == "__main__":

    # Choose and load tire model
    qt_helper = Dialogs(__file__ + 'get TIR')
    filename = str(qt_helper.select_file_dialog(accepted_file_types='*.TIR'))
    coefficients = Mf61
    Mf61.load_model_from_tir(filepath=filename)

    # Create test ranges for slip angles and ratios
    slip_angles = np.linspace(np.deg2rad(-12), np.deg2rad(12), num=100)
    slip_ratios = np.linspace(-0.2, 0.2, num=100)

    # Create forces to plot
    tire_force_data = Mf61.create_tire_plot_test_data(slip_angles, slip_ratios, -0.5 * np.pi/180, 1100, 55158)
    plot_force(tire_force_data)
