'''


Created: 2021
Contributors: Nigel Swab
'''

import warnings
from abc import ABC, abstractmethod
from typing import Union

import numba
import numpy as np
from pandas import Series
from scipy.optimize import curve_fit


# TODO: Should probably add some try : excepts to calculate
# TODO: Add more equations as necessary

# TODO: Sort out what to do with covariance
#  - especially when they can't be estimated
# Temporary fix
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")


class EquationFits(ABC):
    fit_type: str
    fit_parameter: str
    coefficients: np.ndarray
    # covariance: np.ndarray

    @abstractmethod
    def calculate(self, input_data):
        pass

    @abstractmethod
    def fit(self, independent_data, dependent_data):
        pass


class LinearFits(EquationFits):
    def __init__(self, fit_type: str = None, fit_parameter: str = None):
        self.fit_type = fit_type
        self.fit_parameter = fit_parameter
        self.coefficients = np.ndarray
        # self.covariance = np.ndarray

        equations = {'linear1': LinearFits.linear1,
                     'linear2': LinearFits.linear2,
                     'linear3': LinearFits.linear3}

        if self.fit_type in equations:
            self.equation = equations[fit_type]
        else:
            raise AttributeError(f'Fit type {fit_type} not currently included in LinearFits')

    def calculate(self, input_data):
        return self.equation(input_data, *self.coefficients)

    def fit(self, independent_data: Union[Series, list[Series]], dependent_data: Series):
        self.coefficients, _ = curve_fit(self.equation, independent_data, dependent_data)

    @staticmethod
    def linear1(x, p1=0, p2=0):
        return p1 * x + p2

    @staticmethod
    def linear2(x_y, p1=0, p2=0, p3=0):
        x, y = x_y[0], x_y[1]
        return x * p1 + y * p2 + p3

    @staticmethod
    def linear3(x_y_z, p1=0, p2=0, p3=0, p4=0):
        x, y, z = x_y_z[0], x_y_z[1], x_y_z[2]
        return x * p1 + y * p2 * z * p3 + p4


class PolyCurve(EquationFits):
    def __init__(self, fit_type: str = None, fit_parameter: str = None):
        self.fit_type = fit_type
        self.fit_parameter = fit_parameter
        self.coefficients = np.ndarray
        # self.covariance = np.ndarray

        equations = {'poly2': PolyCurve.poly2,
                     'poly3': PolyCurve.poly3,
                     'poly4': PolyCurve.poly4,
                     'poly5': PolyCurve.poly5,
                     'poly6': PolyCurve.poly6}

        if self.fit_type in equations:
            self.equation = equations[fit_type]
        else:
            raise AttributeError(f'Fit type {fit_type} not currently included in PolySurface')

    def calculate(self, input_data):
        return self.equation(input_data, *self.coefficients)

    def fit(self, independent_data: Union[Series, list[Series]], dependent_data: Series):
        self.coefficients, _ = curve_fit(self.equation, independent_data, dependent_data)

    @staticmethod
    def poly2(x, p1=0, p2=0, p3=0):
        return p1 * x ** 2 + p2 * x + p3

    @staticmethod
    def poly3(x, p1=0, p2=0, p3=0, p4=0):
        return p1 * x ** 3 + p2 * x ** 2 + p3 * x + p4

    @staticmethod
    def poly4(x, p1=0, p2=0, p3=0, p4=0, p5=0):
        return p1 * x ** 4 + p2 * x ** 3 + p3 * x ** 2 + p4 * x + p5

    @staticmethod
    def poly5(x, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0):
        return p1 * x ** 5 + p2 * x ** 4 + p3 * x ** 3 + p4 * x ** 2 + p5 * x + p6

    @staticmethod
    def poly6(x, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0):
        return p1 * x ** 6 + p2 * x ** 5 + p3 * x ** 4 + p4 * x ** 3 + p5 * x ** 2 + p6 * x + p7


class PolySurface(EquationFits):
    def __init__(self, fit_type: str, fit_parameter: str = None):
        self.fit_type = fit_type
        self.fit_parameter = fit_parameter
        self.coefficients = np.ndarray
        # self.covariance = np.ndarray

        equations = {'poly12': PolySurface.poly12,
                     'poly14': PolySurface.poly14,
                     'poly22': PolySurface.poly22,
                     'poly23': PolySurface.poly23,
                     'poly32': PolySurface.poly32,
                     'poly33': PolySurface.poly33, }

        if self.fit_type in equations:
            self.equation = equations[fit_type]
        else:
            raise AttributeError(f'Fit type {fit_type} not currently included in PolySurface')

    def calculate(self, input_data):
        return self.equation(input_data, *self.coefficients)

    def fit(self, independent_data: Union[Series, list[Series]], dependent_data: Series):
        self.coefficients, _ = curve_fit(self.equation, independent_data, dependent_data)

    @staticmethod
    def poly12(x_y, p00=0, p10=0, p01=0, p02=0, p11=0):
        # From matlab polynomial surface fit equations
        x, y = x_y[0], x_y[1]
        return p00 + p10 * x + p01 * y + p11 * x * y + p02 * y ** 2

    @staticmethod
    def poly14(x_y, p00=0, p10=0, p01=0, p11=0, p02=0, p12=0, p03=0, p13=0, p04=0):
        x, y = x_y[0], x_y[1]
        return p00 + p10 * x + p01 * y + p11 * x * y + p02 * y ** 2 + p12 * x * y ** 2 + p03 * y ** 3 \
            + p13 * x * y ** 3 + p04 * y ** 4

    @staticmethod
    def poly21(x_y, p00=0, p10=0, p01=0, p20=0, p11=0):
        # From matlab polynomial surface fit equations
        x, y = x_y[0], x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y

    @staticmethod
    def poly22(x_y, p00=0, p10=0, p01=0, p20=0, p11=0, p02=0):
        x, y = x_y[0], x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2


    @staticmethod
    def poly23(x_y, p00=0, p10=0, p01=0, p20=0, p11=0, p02=0, p21=0, p12=0, p03=0) -> np.ndarray:
        # From matlab polynomial surface fit equations
        x, y = x_y[0], x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p21 * x ** 2 * y \
            + p12 * x * y ** 2 + p03 * y * 3

    @staticmethod
    def poly32(x_y, p00=0, p10=0, p01=0, p20=0, p11=0, p02=0, p21=0, p12=0, p30=0) -> np.ndarray:
        # From matlab polynomial surface fit equations
        x, y = x_y[0], x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p30 * x ** 3 \
            + p21 * x ** 2 * y + p12 * x * y ** 2

    @staticmethod
    def poly33(x_y, p00=0, p10=0, p01=0, p20=0, p11=0, p02=0, p21=0, p12=0, p30=0, p03=0) -> np.ndarray:
        # From matlab polynomial surface fit equations
        x, y = x_y[0], x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p30 * x ** 3 \
            + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3
