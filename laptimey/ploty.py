"""A module to allow easy creation of UVFR style plots from simulation data

Created 2022, Contributors: Nigel Swab
"""

from cycler import cycler
from dataclasses import dataclass
import logging
import os
import gc
import numbers
import random
from pathlib import Path
import time
from typing import Union, Optional, Callable
from dataclasses import asdict
import warnings

import plotly.express as px
import matplotlib
import matplotlib.dates as mdates
from matplotlib.figure import Figure, Axes
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from PIL import Image
from scipy import signal
from scipy.optimize import curve_fit

from lib import print_colour
# from plotting import ploty_loader
# from plotting.ploty_input_classes import LimitLine, FitLine
# from tools import fit_data_to_fn

matplotlib.use('Qt5Agg')  # Need to set backend before importing pyplot
from matplotlib import pyplot, style, cm, colors  # noqa:E402(ignore warning)
# warnings.filterwarnings("ignore", message='Starting a Matplotlib GUI outside of the main thread will likely fail')


# class Plotter:
#
#     colours = {'almost black': '#555555',
#                'grey': '#AAAAAA',
#                'white': '#FFFFFF',
#                'vikes blue': '#055EAA',
#                'vikes dark blue': '#003367',
#                'vikes gold': '#FFC528',
#                }
#     line_colours = {'red': '#D32024',
#                     'blue': '#2E9AFE',
#                     'orange': '#FF8000',
#                     'purple': '#AC58FA',
#                     'green': '#04B45F',
#                     'yellow': '#F2F222',
#                     'pink': '#EB3BB6',
#                     'cyan': '#27F2D7',
#                     'dark purple': '#4B18CC',
#                     'light green': '#9CED21'}
#
#     def __init__(
#             self,
#             data: Union[DataFrame, str, Path, list[Union[DataFrame, str, Path]]],
#             # metadata: Union[list[dict], dict] = None,
#             # split_dataset: bool = False,  # Split data into passes for visualization
#             # selected_datasets: list = None,  # Choose which passes or runs to plot
#             # polar: bool = False,  # Switch from Cartesian to Polar coordinates
#             embedded_fig: tuple[Figure, list[Axes]] = False  # For figures embedded in a gui
#     ):
#         """Creates a properly formatted blank UVFR chart (legend formatted later)
#          - Use one instance per figure
#         """
#         pass

class PlotDataManipulation:

    def split_data(self,
                   data: Union[df, list[df]],
                   split_param: str,
                   ) -> list[df]:
        """
        Splits passes into separate Dataframes

        Args:
            data ():
            split_param ():

        Returns:

        """
        pass

    def sort_data(self,
                  data: Union[df, list[df]],
                  params: Union[str, list[str]],
                  inplace: bool = False,
                  ) -> Optional[Union[df, list[df]]]:
        pass

    def extract_const_slip_lines(self,
                                 data: df,
                                 ) -> list[df]:
        pass

    def extract_const_steer_lines(self,
                                  data: df,
                                  ) -> list[df]:
        pass


if __name__ == "__main__":
    data_sheet = r"C:\Users\Nigel Swab\Desktop\sample_data.csv"
