"""
Module to aid with unit conversions specific to this repository

Created: 2021
Contributors: Nigel Swab
"""
from typing import Union
from numpy import pi
from numba import vectorize, float64


@vectorize([float64(float64)], nopython=True)
def psi_to_pa(psi: Union[float, list[float]]) -> Union[float, list[float]]:
    return psi * 6894.76


@vectorize([float64(float64)], nopython=True)
def pa_to_psi(pa: Union[float, list[float]]) -> Union[float, list[float]]:
    return pa / 6894.76


@vectorize([float64(float64)], nopython=True)
def mps2_to_g(mps2: Union[float, list[float]]) -> Union[float, list[float]]:
    return mps2 / 9.80665


@vectorize([float64(float64)], nopython=True)
def g_to_mps2(g: Union[float, list[float]]) -> Union[float, list[float]]:
    return g * 9.80665


@vectorize([float64(float64)], nopython=True)
def lbpin_to_npm(lbpin: Union[float, list[float]]) -> Union[float, list[float]]:
    return lbpin * 175.126835


@vectorize([float64(float64)], nopython=True)
def npm_to_lbpin(npm: Union[float, list[float]]) -> Union[float, list[float]]:
    return npm / 175.126835


@vectorize([float64(float64)], nopython=True)
def in_to_m(inches: Union[float, list[float]]) -> Union[float, list[float]]:
    return inches * 0.0254


@vectorize([float64(float64)], nopython=True)
def m_to_in(m: Union[float, list[float]]) -> Union[float, list[float]]:
    return m / 0.0254


@vectorize([float64(float64)], nopython=True)
def lb_to_kg(lb: Union[float, list[float]]) -> Union[float, list[float]]:
    return lb * 0.453592


@vectorize([float64(float64)], nopython=True)
def kg_to_lb(kg: Union[float, list[float]]) -> Union[float, list[float]]:
    return kg / 0.453592


@vectorize([float64(float64)], nopython=True)
def kph_to_mps(kph: Union[float, list[float]]) -> Union[float, list[float]]:
    return kph * 0.277778


@vectorize([float64(float64)], nopython=True)
def mps_to_kph(mps: Union[float, list[float]]) -> Union[float, list[float]]:
    return mps / 0.277778


@vectorize([float64(float64)], nopython=True)
def mmprev_to_mmprad(mmprev: Union[float, list[float]]) -> Union[float, list[float]]:
    return mmprev / (2 * pi)


@vectorize([float64(float64)], nopython=True)
def mmprad_to_mmprev(mmprad: Union[float, list[float]]) -> Union[float, list[float]]:
    return mmprad * 2 * pi


if __name__ == "__main__":
    print(kg_to_lb([20, 40]))
