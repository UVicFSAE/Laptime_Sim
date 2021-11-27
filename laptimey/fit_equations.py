'''


Contributors: Nigel Swab
Created: 2021
'''

import numpy as np


class PolySurface:
    def __init__(self, fit_type: str = ''):
        self.fit_type = fit_type

        pass

    def calculate(self, coefficients):
        return eval(self.fit_type)

    @staticmethod
    def poly12(x_y, p00=0, p10=0, p01=0, p02=0, p11=0):
        # From matlab polynomial surface fit equations
        x = x_y[0]
        y = x_y[1]
        return p00 + p10*x + p01*y + p11*x*y + p02 * y ** 2

    @staticmethod
    def poly21(x_y, p00=0, p10=0, p01=0, p20=0, p11=0):
        # From matlab polynomial surface fit equations
        x = x_y[0]
        y = x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y

    @staticmethod
    def poly22(x_y, p00=0, p10=0, p01=0, p20=0, p11=0, p02=0):
        x = x_y[0]
        y = x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2

    @staticmethod
    def poly23(x_y, p00=0, p10=0, p01=0, p20=0, p11=0, p02=0, p21=0, p12=0, p03=0) -> np.ndarray:
        # From matlab polynomial surface fit equations
        x = x_y[0]
        y = x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p21 * x ** 2 * y \
            + p12 * x * y ** 2 + p03 * y * 3

    @staticmethod
    def poly32(x_y, p00=0, p10=0, p01=0, p20=0, p11=0, p02=0, p21=0, p12=0, p30=0) -> np.ndarray:
        # From matlab polynomial surface fit equations
        x = x_y[0]
        y = x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p30 * x ** 3 \
            + p21 * x ** 2 * y + p12 * x * y ** 2

    @staticmethod
    def poly33(x_y, p00=0, p10=0, p01=0, p20=0, p11=0, p02=0, p21=0, p12=0, p30=0, p03=0) -> np.ndarray:
        # From matlab polynomial surface fit equations
        x = x_y[0]
        y = x_y[1]
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02*y ** 2 + p30 * x ** 3 \
            + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3