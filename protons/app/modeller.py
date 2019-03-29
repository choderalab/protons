# coding=utf-8
"""Pre-configured OpenMM Modeller object for use with the default protons forcefield. Modified from original source code by Peter Eastman."""

from .topology import Topology
from simtk.openmm.app import modeller
from simtk.openmm.vec3 import Vec3
from simtk.openmm import (
    System,
    Context,
    CustomNonbondedForce,
    HarmonicBondForce,
    HarmonicAngleForce,
    VerletIntegrator,
    LocalEnergyMinimizer,
)
from simtk.unit import nanometer, degree, acos, dot, norm
import os
from . import element as elem
import random
from copy import deepcopy


PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
# Topology.loadBondDefinitions(os.path.join(PACKAGE_ROOT, 'data', 'bonds-amber10-constph.xml'))


class Modeller(modeller.Modeller):
    """A modification of the simtk.openmm.app Modeller class.
    Credits
    -------
    The original class was written by Peter Eastman.
    """

    @classmethod
    def unloadHydrogenDefinitions(cls):
        """Unloading hydrogen definitions."""
        cls._residueHydrogens = {}
        cls._hasLoadedStandardHydrogens = False

    def addHydrogens(self, forcefield=None, pH=None, variants=None, platform=None):
        """Add missing hydrogens to the model.
        This function automatically changes compatible residues into their constant-pH variant if no variant is specified.:
        Aspartic acid:
            AS4: Form with a 2 hydrogens on each one of the delta oxygens (syn,anti)
                It has 5 titration states.
            Alternative:
            AS2: Has 2 hydrogens (syn, anti) on one of the delta oxygens
                It has 3 titration states.
        Cysteine:
            CYS: Neutral form with a hydrogen on the sulfur
            CYX: No hydrogen on the sulfur (either negatively charged, or part of a disulfide bond)
        Glutamic acid:
            GL4: Form with a 2 hydrogens on each one of the epsilon oxygens (syn,anti)
                It has 5 titration states.
        Histidine:
            HIP: Positively charged form with hydrogens on both ND1 and NE2
                It has 3 titration states.
        The variant to use for each residue is determined by the following rules:
        1. Any Cysteine that participates in a disulfide bond uses the CYX variant regardless of pH.
        2. Other residues are all set to maximally protonated state, which can be updated using a proton drive
        You can override these rules by explicitly specifying a variant for any residue.  To do that, provide a list for the
        'variants' parameter, and set the corresponding element to the name of the variant to use.
        A special case is when the model already contains a hydrogen that should not be present in the desired variant.
        If you explicitly specify a variant using the 'variants' parameter, the residue will be modified to match the
        desired variant, removing hydrogens if necessary.  On the other hand, for residues whose variant is selected
        automatically, this function will only add hydrogens.  It will never remove ones that are already present in the
        model.
        Definitions for standard amino acids and nucleotides are built in.  You can call loadHydrogenDefinitions() to load
        additional definitions for other residue types.
        Parameters
        ----------
        forcefield : ForceField=None
            the ForceField to use for determining the positions of hydrogens.
            If this is None, positions will be picked which are generally
            reasonable but not optimized for any particular ForceField.
        pH : None,
            Kept for compatibility reasons. Has no effect.
        variants : list=None
            an optional list of variants to use.  If this is specified, its
            length must equal the number of residues in the model.  variants[i]
            is the name of the variant to use for residue i (indexed starting at
            0). If an element is None, the standard rules will be followed to
            select a variant for that residue.
        platform : Platform=None
            the Platform to use when computing the hydrogen atom positions.  If
            this is None, the default Platform will be used.
        Returns
        -------
        list
             a list of what variant was actually selected for each residue,
             in the same format as the variants parameter
        Notes
        -----
        This function does not use a pH specification. The argument is kept for compatibility reasons.
        """
        # Check the list of variants.

        if pH is not None:
            print("Ignored pH argument provided for constant-pH residues.")

        residues = list(self.topology.residues())
        if variants is not None:
            if len(variants) != len(residues):
                raise ValueError(
                    "The length of the variants list must equal the number of residues"
                )
        else:
            variants = [None] * len(residues)
        actualVariants = [None] * len(residues)

        # Load the residue specifications.

        if not Modeller._hasLoadedStandardHydrogens:
            Modeller.loadHydrogenDefinitions(
                os.path.join(
                    os.path.dirname(__file__), "data", "hydrogens-amber10-constph.xml"
                )
            )

        # Make a list of atoms bonded to each atom.

        bonded = {}
        for atom in self.topology.atoms():
            bonded[atom] = []
        for atom1, atom2 in self.topology.bonds():
            bonded[atom1].append(atom2)
            bonded[atom2].append(atom1)

        # Define a function that decides whether a set of atoms form a hydrogen bond, using fairly tolerant criteria.

        def isHbond(d, h, a):
            if norm(d - a) > 0.35 * nanometer:
                return False
            deltaDH = h - d
            deltaHA = a - h
            deltaDH /= norm(deltaDH)
            deltaHA /= norm(deltaHA)
            return acos(dot(deltaDH, deltaHA)) < 50 * degree

        # Loop over residues.

        newTopology = Topology()
        newTopology.setPeriodicBoxVectors(self.topology.getPeriodicBoxVectors())
        newAtoms = {}
        newPositions = [] * nanometer
        newIndices = []
        acceptors = [
            atom
            for atom in self.topology.atoms()
            if atom.element in (elem.oxygen, elem.nitrogen)
        ]
        for chain in self.topology.chains():
            newChain = newTopology.addChain(chain.id)
            for residue in chain.residues():
                newResidue = newTopology.addResidue(residue.name, newChain, residue.id)
                isNTerminal = residue == chain._residues[0]
                isCTerminal = residue == chain._residues[-1]
                if residue.name in Modeller._residueHydrogens:
                    # Add hydrogens.  First select which variant to use.

                    spec = Modeller._residueHydrogens[residue.name]
                    variant = variants[residue.index]
                    if variant is None:
                        if residue.name == "CYS":
                            # If this is part of a disulfide, use CYX.

                            sulfur = [
                                atom
                                for atom in residue.atoms()
                                if atom.element == elem.sulfur
                            ]
                            if len(sulfur) == 1 and any(
                                (atom.residue != residue for atom in bonded[sulfur[0]])
                            ):
                                variant = "CYX"
                        if residue.name == "HIS":
                            variant = "HIP"
                        if residue.name == "GLU":
                            variant = "GL4"
                        if residue.name == "ASP":
                            variant = "AS4"
                    if variant is not None and variant not in spec.variants:
                        raise ValueError(
                            "Illegal variant for %s residue: %s"
                            % (residue.name, variant)
                        )
                    actualVariants[residue.index] = variant
                    removeExtraHydrogens = variants[residue.index] is not None

                    # Make a list of hydrogens that should be present in the residue.

                    parents = [
                        atom
                        for atom in residue.atoms()
                        if atom.element != elem.hydrogen
                    ]
                    parentNames = [atom.name for atom in parents]
                    hydrogens = [
                        h
                        for h in spec.hydrogens
                        if (variant is None)
                        or (h.variants is None)
                        or (h.variants is not None and variant in h.variants)
                    ]
                    hydrogens = [
                        h
                        for h in hydrogens
                        if h.terminal is None
                        or (isNTerminal and h.terminal == "N")
                        or (isCTerminal and h.terminal == "C")
                    ]
                    hydrogens = [h for h in hydrogens if h.parent in parentNames]

                    # Loop over atoms in the residue, adding them to the new topology along with required hydrogens.
                    for parent in residue.atoms():
                        # Check whether this is a hydrogen that should be removed.

                        if (
                            removeExtraHydrogens
                            and parent.element == elem.hydrogen
                            and not any(parent.name == h.name for h in hydrogens)
                        ):
                            continue

                        # Add the atom.
<<<<<<< HEAD
                        newAtom = newTopology.addAtom(parent.name, parent.element, newResidue)
=======

                        newAtom = newTopology.addAtom(
                            parent.name, parent.element, newResidue
                        )
>>>>>>> develop
                        newAtoms[parent] = newAtom
                        newPositions.append(deepcopy(self.positions[parent.index]))
                        if parent in parents:
                            # Match expected hydrogens with existing ones and find which ones need to be added.

                            existing = [
                                atom
                                for atom in bonded[parent]
                                if atom.element == elem.hydrogen
                            ]
                            expected = [h for h in hydrogens if h.parent == parent.name]
                            if len(existing) < len(expected):
                                # Try to match up existing hydrogens to expected ones.

                                matches = []
                                for e in existing:
                                    match = [h for h in expected if h.name == e.name]
                                    if len(match) > 0:
                                        matches.append(match[0])
                                        expected.remove(match[0])
                                    else:
                                        matches.append(None)

                                # If any hydrogens couldn't be matched by name, just match them arbitrarily.

                                for i in range(len(matches)):
                                    if matches[i] is None:
                                        matches[i] = expected[-1]
                                        expected.remove(expected[-1])

                                # Add the missing hydrogens.

                                for h in expected:
                                    newH = newTopology.addAtom(
                                        h.name, elem.hydrogen, newResidue
                                    )
                                    newIndices.append(newH.index)
                                    delta = Vec3(0, 0, 0) * nanometer
                                    if len(bonded[parent]) > 0:
                                        for other in bonded[parent]:
                                            delta += (
                                                self.positions[parent.index]
                                                - self.positions[other.index]
                                            )
                                    else:
                                        delta = (
                                            Vec3(
                                                random.random(),
                                                random.random(),
                                                random.random(),
                                            )
                                            * nanometer
                                        )
                                    delta *= 0.1 * nanometer / norm(delta)
                                    delta += (
                                        0.05
                                        * Vec3(
                                            random.random(),
                                            random.random(),
                                            random.random(),
                                        )
                                        * nanometer
                                    )
                                    delta *= 0.1 * nanometer / norm(delta)
                                    newPositions.append(
                                        self.positions[parent.index] + delta
                                    )
                                    newTopology.addBond(newAtom, newH)
                else:
                    # Just copy over the residue.

                    for atom in residue.atoms():
                        newAtom = newTopology.addAtom(
                            atom.name, atom.element, newResidue
                        )
                        newAtoms[atom] = newAtom
                        newPositions.append(deepcopy(self.positions[atom.index]))
        for bond in self.topology.bonds():
            if bond[0] in newAtoms and bond[1] in newAtoms:
                newTopology.addBond(newAtoms[bond[0]], newAtoms[bond[1]])

        # The hydrogens were added at random positions.  Now perform an energy minimization to fix them up.

        if forcefield is not None:
            # Use the ForceField the user specified.

            system = forcefield.createSystem(newTopology, rigidWater=False)
            atoms = list(newTopology.atoms())
            for i in range(system.getNumParticles()):
                if atoms[i].element != elem.hydrogen:
                    # This is a heavy atom, so make it immobile.
                    system.setParticleMass(i, 0)
        else:
            # Create a System that restrains the distance of each hydrogen from its parent atom
            # and causes hydrogens to spread out evenly.

            system = System()
            nonbonded = CustomNonbondedForce("100/((r/0.1)^4+1)")
            bonds = HarmonicBondForce()
            angles = HarmonicAngleForce()
            system.addForce(nonbonded)
            system.addForce(bonds)
            system.addForce(angles)
            bondedTo = []
            for atom in newTopology.atoms():
                nonbonded.addParticle([])
                if atom.element != elem.hydrogen:
                    system.addParticle(0.0)
                else:
                    system.addParticle(1.0)
                bondedTo.append([])
            for atom1, atom2 in newTopology.bonds():
                if atom1.element == elem.hydrogen or atom2.element == elem.hydrogen:
                    bonds.addBond(atom1.index, atom2.index, 0.1, 100_000.0)
                bondedTo[atom1.index].append(atom2)
                bondedTo[atom2.index].append(atom1)
            for residue in newTopology.residues():
                if residue.name == "HOH":
                    # Add an angle term to make the water geometry correct.

                    atoms = list(residue.atoms())
                    oindex = [
                        i for i in range(len(atoms)) if atoms[i].element == elem.oxygen
                    ]
                    if len(atoms) == 3 and len(oindex) == 1:
                        hindex = list(set([0, 1, 2]) - set(oindex))
                        angles.addAngle(
                            atoms[hindex[0]].index,
                            atoms[oindex[0]].index,
                            atoms[hindex[1]].index,
                            1.824,
                            836.8,
                        )
                else:
                    # Add angle terms for any hydroxyls.

                    for atom in residue.atoms():
                        index = atom.index
                        if (
                            atom.element == elem.oxygen
                            and len(bondedTo[index]) == 2
                            and elem.hydrogen in (a.element for a in bondedTo[index])
                        ):
                            angles.addAngle(
                                bondedTo[index][0].index,
                                index,
                                bondedTo[index][1].index,
                                1.894,
                                460.24,
                            )

        if platform is None:
            context = Context(system, VerletIntegrator(0.0))
        else:
            context = Context(system, VerletIntegrator(0.0), platform)
        context.setPositions(newPositions)
        LocalEnergyMinimizer.minimize(context, 1.0, 50)
        self.topology = newTopology
        self.positions = context.getState(getPositions=True).getPositions()
        del context
        return actualVariants


Modeller.loadHydrogenDefinitions(
    os.path.join(PACKAGE_ROOT, "data", "hydrogens-amber10-constph.xml")
)
