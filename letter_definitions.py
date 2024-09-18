#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:03:51 2024

@author: george
"""

import numpy as np

def create_letter_A():
    vertices = np.array([
        [0, 0, 0], [4, 0, 0], [0, 0, 2], [4, 0, 2],
        [14, 0, 0], [18, 0, 0], [14, 0, 2], [18, 0, 2],
        [5, 6, 0], [13, 6, 0], [5, 6, 2], [13, 6, 2],
        [7, 13, 0], [11, 13, 0], [7, 13, 2], [11, 13, 2],
        [9, 17, 0], [9, 17, 2], [7, 24, 0], [11, 24, 0],
        [7, 24, 2], [11, 24, 2]
    ])
    faces = np.array([
        [0, 1, 3, 2], [4, 5, 7, 6], [8, 9, 11, 10],
        [12, 13, 15, 14], [18, 19, 21, 20],
        [0, 2, 20, 18], [5, 7, 21, 19], [1, 3, 10, 8],
        [4, 6, 11, 9], [12, 14, 17, 16], [13, 15, 17, 16]
    ])
    return vertices, faces

def create_letter_B():
    vertices = np.array([
        [0, 0, 0], [0, 24, 0], [0, 0, 2], [0, 24, 2],
        [12, 0, 0], [12, 12, 0], [12, 24, 0],
        [12, 0, 2], [12, 12, 2], [12, 24, 2],
        [14, 2, 0], [14, 10, 0], [14, 14, 0], [14, 22, 0],
        [14, 2, 2], [14, 10, 2], [14, 14, 2], [14, 22, 2]
    ])
    faces = np.array([
        [0, 1, 3, 2], [0, 4, 7, 2], [1, 6, 9, 3],
        [4, 5, 8, 7], [5, 6, 9, 8],
        [4, 10, 14, 7], [5, 12, 16, 8],
        [10, 11, 15, 14], [12, 13, 17, 16],
        [11, 5, 8, 15], [13, 6, 9, 17]
    ])
    return vertices, faces

# Add more letter definitions here...

def create_number_0():
    vertices = np.array([
        [0, 0, 0], [12, 0, 0], [0, 24, 0], [12, 24, 0],
        [0, 0, 2], [12, 0, 2], [0, 24, 2], [12, 24, 2]
    ])
    faces = np.array([
        [0, 1, 3, 2], [4, 5, 7, 6],
        [0, 2, 6, 4], [1, 3, 7, 5],
        [0, 1, 5, 4], [2, 3, 7, 6]
    ])
    return vertices, faces

# Add more number definitions here...

def create_symbol_exclamation():
    vertices = np.array([
        [0, 20, 0], [2, 20, 0], [0, 24, 0], [2, 24, 0],
        [0, 20, 2], [2, 20, 2], [0, 24, 2], [2, 24, 2],
        [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
        [0, 0, 2], [2, 0, 2], [0, 2, 2], [2, 2, 2]
    ])
    faces = np.array([
        [0, 1, 3, 2], [4, 5, 7, 6],
        [0, 2, 6, 4], [1, 3, 7, 5],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [8, 9, 11, 10], [12, 13, 15, 14],
        [8, 10, 14, 12], [9, 11, 15, 13],
        [8, 9, 13, 12], [10, 11, 15, 14]
    ])
    return vertices, faces

# Add more symbol definitions here...

LETTER_DEFINITIONS = {
    'A': create_letter_A(),
    'B': create_letter_B(),
    # Add more letters here...
}

NUMBER_DEFINITIONS = {
    '0': create_number_0(),
    # Add more numbers here...
}

SYMBOL_DEFINITIONS = {
    '!': create_symbol_exclamation(),
    # Add more symbols here...
}

# Combine all definitions
ALL_CHARACTERS = {**LETTER_DEFINITIONS, **NUMBER_DEFINITIONS, **SYMBOL_DEFINITIONS}
