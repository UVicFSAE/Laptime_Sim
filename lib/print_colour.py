"""
Created on Jul 31, 2019

@Author: npaolini

A function to print to console in colour
"""

import sys


def printy(the_text, colour: str = 'pale yellow', bold: bool = False, end: str = '\n') -> None:
    """
    Prints text to console in colour to make it stand out (Don't overuse)

    :param end: Character to put at end of string (typically '\n', or '')
    :param the_text: Text string to be printed
    :param colour: Colour matching one of the conditions below
    :param bold: Make the font bold
    """

    bold = 1 if bold else 0

    if colour.lower() == 're_raise_yellow':
        # Use this if you will be raising an exception after using print_colour
        # The colour is bright to distinguish from red exception text and the pause stops console
        # statements from getting out of order
        sys.stdout.flush()
        print(f'\x1b[{bold};93m{the_text}\x1b[0m', end=end)

    elif colour.lower() == 'red':
        print(f'\x1b[{bold};31m{the_text}\x1b[0m', end=end)

    elif colour.lower() == 'yellow':
        print(f'\x1b[{bold};32m{the_text}\x1b[0m', end=end)

    elif colour.lower() == 'pale yellow':
        print(f'\x1b[{bold};33m{the_text}\x1b[0m', end=end)

    elif colour.lower() == 'blue':
        print(f'\x1b[{bold};34m{the_text}\x1b[0m', end=end)

    elif colour.lower() == 'purple':
        print(f'\x1b[{bold};35m{the_text}\x1b[0m', end=end)

    elif colour.lower() == 'turquoise':
        print(f'\x1b[{bold};36m{the_text}\x1b[0m', end=end)
    else:
        print(the_text)
