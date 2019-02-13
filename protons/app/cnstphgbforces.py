"""
A recreation of the various GB variants implemented via CustomGBForce

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012 University of Virginia and the Authors.
Authors: Christoph Klein, Michael R. Shirts
Contributors: Jason M. Swails

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# JMS: This is a copy of simtk/openmm/app/internal/customgbforces.py but
# adjusted to exclude atoms with a charge of 0. This is common in constant pH
# for protons that are "off", or dummy atoms at the time. These exclusions are
# implemented by multiplying "I" by step(abs(q1)-1e-8) * step(abs(q2)-1e-8). The
# product of the steps will be 0 only if the absolute value of the charge is
# <1e-8, so it should work for dummy atoms. Overall, this is a hack.

from __future__ import division

from simtk.openmm import CustomGBForce
from simtk.openmm.app.internal.customgbforces import d0, m0, _createEnergyTerms

"""
Amber Equivalent: igb = 1
"""


def GBSAHCTForce(solventDielectric=78.5, soluteDielectric=1, SA=None, cutoff=None):

    custom = CustomGBForce()

    custom.addPerParticleParameter("q")
    custom.addPerParticleParameter("radius")
    custom.addPerParticleParameter("scale")
    custom.addGlobalParameter("solventDielectric", solventDielectric)
    custom.addGlobalParameter("soluteDielectric", soluteDielectric)
    custom.addGlobalParameter("offset", 0.009)
    custom.addComputedValue(
        "I",
        "step(r+sr2-or1)*excl*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
        "excl=step(abs(q1)-0.00000001)*step(abs(q2)-0.00000001);"  # exclude pair where one atom is not charged
        "U=r+sr2;"
        "L=max(or1, D);"
        "D=abs(r-sr2);"
        "sr2 = scale2*or2;"
        "or1 = radius1-offset; or2 = radius2-offset",
        CustomGBForce.ParticlePairNoExclusions,
    )

    custom.addComputedValue(
        "B", "1/(1/or-I);" "or=radius-offset", CustomGBForce.SingleParticle
    )
    _createEnergyTerms(custom, SA, cutoff)
    return custom


"""
Amber Equivalents: igb = 2
"""


def GBSAOBC1Force(solventDielectric=78.5, soluteDielectric=1, SA=None, cutoff=None):

    custom = CustomGBForce()

    custom.addPerParticleParameter("q")
    custom.addPerParticleParameter("radius")
    custom.addPerParticleParameter("scale")
    custom.addGlobalParameter("solventDielectric", solventDielectric)
    custom.addGlobalParameter("soluteDielectric", soluteDielectric)
    custom.addGlobalParameter("offset", 0.009)
    custom.addComputedValue(
        "I",
        "step(r+sr2-or1)*excl*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
        "excl=step(abs(q1)-0.00000001)*step(abs(q2)-0.00000001);"  # exclude pair where one atom is not charged
        "U=r+sr2;"
        "L=max(or1, D);"
        "D=abs(r-sr2);"
        "sr2 = scale2*or2;"
        "or1 = radius1-offset; or2 = radius2-offset",
        CustomGBForce.ParticlePairNoExclusions,
    )

    custom.addComputedValue(
        "B",
        "1/(1/or-tanh(0.8*psi+2.909125*psi^3)/radius);" "psi=I*or; or=radius-offset",
        CustomGBForce.SingleParticle,
    )
    _createEnergyTerms(custom, SA, cutoff)
    return custom


"""
Amber Equivalents: igb = 5
"""


def GBSAOBC2Force(solventDielectric=78.5, soluteDielectric=1, SA=None, cutoff=None):

    custom = CustomGBForce()

    custom.addPerParticleParameter("q")
    custom.addPerParticleParameter("radius")
    custom.addPerParticleParameter("scale")
    custom.addGlobalParameter("solventDielectric", solventDielectric)
    custom.addGlobalParameter("soluteDielectric", soluteDielectric)
    custom.addGlobalParameter("offset", 0.009)
    custom.addComputedValue(
        "I",
        "step(r+sr2-or1)*excl*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
        "excl=step(abs(q1)-0.00000001)*step(abs(q2)-0.00000001);"  # exclude pair where one atom is not charged
        "U=r+sr2;"
        "L=max(or1, D);"
        "D=abs(r-sr2);"
        "sr2 = scale2*or2;"
        "or1 = radius1-offset; or2 = radius2-offset",
        CustomGBForce.ParticlePairNoExclusions,
    )

    custom.addComputedValue(
        "B",
        "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);" "psi=I*or; or=radius-offset",
        CustomGBForce.SingleParticle,
    )
    _createEnergyTerms(custom, SA, cutoff)
    return custom


"""
Amber Equivalents: igb = 7
"""


def GBSAGBnForce(solventDielectric=78.5, soluteDielectric=1, SA=None, cutoff=None):

    """
    Indexing for tables:
        input: radius1, radius2
        index = (radius2*200-20)*21 + (radius1*200-20)
        output: index of desired value in row-by-row, 1D version of Tables 3 & 4
    """

    custom = CustomGBForce()

    custom.addPerParticleParameter("q")
    custom.addPerParticleParameter("radius")
    custom.addPerParticleParameter("scale")

    custom.addGlobalParameter("solventDielectric", solventDielectric)
    custom.addGlobalParameter("soluteDielectric", soluteDielectric)
    custom.addGlobalParameter("offset", 0.009)
    custom.addGlobalParameter("neckScale", 0.361825)
    custom.addGlobalParameter("neckCut", 0.68)

    custom.addFunction("getd0", d0, 0, 440)
    custom.addFunction("getm0", m0, 0, 440)

    custom.addComputedValue(
        "I",
        "Ivdw+neckScale*Ineck;"
        "Ineck=step(radius1+radius2+neckCut-r)*getm0(index)/(1+100*(r-getd0(index))^2+0.3*1000000*(r-getd0(index))^6);"
        "index = (radius2*200-20)*21 + (radius1*200-20);"
        "Ivdw=step(r+sr2-or1)*excl*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
        "excl=step(abs(q1)-0.00000001)*step(abs(q2)-0.00000001);"  # exclude pair where one atom is not charged
        "U=r+sr2;"
        "L=max(or1, D);"
        "D=abs(r-sr2);"
        "sr2 = scale2*or2;"
        "or1 = radius1-offset; or2 = radius2-offset",
        CustomGBForce.ParticlePairNoExclusions,
    )

    custom.addComputedValue(
        "B",
        "1/(1/or-tanh(1.09511284*psi-1.907992938*psi^2+2.50798245*psi^3)/radius);"
        "psi=I*or; or=radius-offset",
        CustomGBForce.SingleParticle,
    )
    _createEnergyTerms(custom, SA, cutoff)
    return custom


"""
Amber Equivalents: igb = 8
"""


def GBSAGBn2Force(solventDielectric=78.5, soluteDielectric=1, SA=None, cutoff=None):

    """
    Indexing for tables:
        input: radius1, radius2
        index = (radius2*200-20)*21 + (radius1*200-20)
        output: index of desired value in row-by-row, 1D version of Tables 3 & 4
    """

    custom = CustomGBForce()

    custom.addPerParticleParameter("q")
    custom.addPerParticleParameter("radius")
    custom.addPerParticleParameter("scale")
    custom.addPerParticleParameter("alpha")
    custom.addPerParticleParameter("beta")
    custom.addPerParticleParameter("gamma")

    custom.addGlobalParameter("solventDielectric", solventDielectric)
    custom.addGlobalParameter("soluteDielectric", soluteDielectric)
    custom.addGlobalParameter("offset", 0.0195141)
    custom.addGlobalParameter("neckScale", 0.826836)
    custom.addGlobalParameter("neckCut", 0.68)

    custom.addFunction("getd0", d0, 0, 440)
    custom.addFunction("getm0", m0, 0, 440)

    custom.addComputedValue(
        "I",
        "Ivdw+neckScale*Ineck;"
        "Ineck=step(radius1+radius2+neckCut-r)*getm0(index)/(1+100*(r-getd0(index))^2+0.3*1000000*(r-getd0(index))^6);"
        "index = (radius2*200-20)*21 + (radius1*200-20);"
        "Ivdw=step(r+sr2-or1)*excl*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
        "excl=step(abs(q1)-0.00000001)*step(abs(q2)-0.00000001);"  # exclude pair where one atom is not charged
        "U=r+sr2;"
        "L=max(or1, D);"
        "D=abs(r-sr2);"
        "sr2 = scale2*or2;"
        "or1 = radius1-offset; or2 = radius2-offset",
        CustomGBForce.ParticlePairNoExclusions,
    )

    custom.addComputedValue(
        "B",
        "1/(1/or-tanh(alpha*psi-beta*psi^2+gamma*psi^3)/radius);"
        "psi=I*or; or=radius-offset",
        CustomGBForce.SingleParticle,
    )
    _createEnergyTerms(custom, SA, cutoff)
    return custom


def register():
    """
    Replaces simtk.openmm.app.internal.customgbforce forces with the ones
    defined in this module. This simplifies the process of generating a system
    using the existing API
    """
    import simtk.openmm.app.internal.customgbforces as other

    other.GBSAHCTForce = GBSAHCTForce
    other.GBSAOBC1Force = GBSAOBC1Force
    other.GBSAOBC2Force = GBSAOBC2Force
    other.GBSAGBnForce = GBSAGBnForce
    other.GBSAGBn2Force = GBSAGBn2Force
