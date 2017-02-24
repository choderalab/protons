# coding=utf-8
"""Automatically linked absolute paths to forcefield files included with the library."""

from . import get_data

protonsff = get_data('protons.xml', 'forcefields')
hydrogens = get_data('hydrogens-protons.xml', 'forcefields')
bonds = get_data('bonds-protons.xml', 'forcefields')
ions_spce = get_data('ions_spce.xml', 'forcefields')
ions_tip3p = get_data('ions_tip3p.xml', 'forcefields')
ions_tip4pew = get_data('ions_tip4pew.xml', 'forcefields')
gaff = get_data('gaff.xml', 'forcefields')
gaff2 = get_data('gaff2.xml', 'forcefields')
