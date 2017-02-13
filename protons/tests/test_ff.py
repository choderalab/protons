# coding=utf-8
"""Test the reading of forcefield files included with the package"""

from protons import ff
from simtk.openmm import app


def test_reading_protons():
    """Read parameters and templates protons.xml using OpenMM."""
    parsed = app.ForceField(ff.protons)


def test_reading_hydrogens():
    """Read hydrogen definitions in hydrogens-protons.xml using OpenMM."""
    parsed = app.Modeller.loadHydrogenDefinitions(ff.hydrogens)


def test_reading_ions_spce():
    """Read parameters and templates in ions_spce.xml using OpenMM."""
    parsed = app.ForceField(ff.ions_spce)


def test_reading_ions_tip3p():
    """Read parameters and templates in ions_tip3p.xml using OpenMM."""
    parsed = app.ForceField(ff.ions_tip3p)


def test_reading_ions_tip4pew():
    """Read parameters and templates in ions_tip4pew.xml using OpenMM."""
    parsed = app.ForceField(ff.ions_tip4pew)


def test_reading_gaff():
    """Read parameters and templates in gaff.xml using OpenMM."""
    parsed = app.ForceField(ff.gaff)


def test_reading_gaff2():
    """Read parameters and templates in gaff2.xml using OpenMM."""
    parsed = app.ForceField(ff.gaff2)

