from __future__ import print_function
from builtins import input # python 2 and 3 compatible commandline input
from glob import glob
import openmoltools as omt
from joblib import Parallel, delayed
from lxml import etree, objectify
import re, os, sys, logging, tempfile, shutil, random
from collections import OrderedDict
from math import exp
from constph.logger import logger


def _make_xml_root(rootname):
    """
    Create a new xmlfile to store epik data for each state

    Returns
    -------
        Xml tree
    """
    xml = '<{0}></{0}>'.format(rootname)
    root = objectify.fromstring(xml)
    return root


def _make_cph_xml():
    """
    Returns an empty CPH xml file template
    """

    ff = _make_xml_root("ForceField")
    iso = _make_xml_root("IsomerData")
    atypes = _make_xml_root("AtomTypes")
    ress = _make_xml_root("Residues")
    resi = _make_xml_root("Residue")
    hbf = _make_xml_root("HarmonicBondForce")
    haf = _make_xml_root("HarmonicAngleForce")
    ptf = _make_xml_root("PeriodicTorsionForce")
    resi.append(iso)
    ress.append(resi)
    ff.append(atypes)
    ff.append(ress)
    ff.append(hbf)
    ff.append(haf)
    ff.append(ptf)

    return ff


class _Bond(object):
    """
    Private class representing a bond between two atoms. Supports comparisons.
    """
    def __init__(self, atom1, atom2):
        atom1 = int(atom1)
        atom2 = int(atom2)
        if atom1 < atom2:
            self.from_atom = atom1
            self.to_atom = atom2
        elif atom2 < atom1:
            self.from_atom = atom2
            self.to_atom = atom1
        else:
            raise ValueError("Can't define a bond from one atom to itself!")

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __gt__(self, other):
        if self.from_atom > other.from_atom:
            return True
        elif self.from_atom == other.from_atom and self.to_atom > other.to_atom:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.from_atom < other.from_atom:
            return True
        elif self.from_atom == other.from_atom and self.to_atom < other.to_atom:
            return True
        else:
            return False

    def __le__(self, other):
        if self.from_atom < other.from_atom:
            return True
        elif self.from_atom == other.from_atom and self.to_atom <= other.to_atom:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.from_atom > other.from_atom:
            return True
        elif self.from_atom == other.from_atom and self.to_atom >= other.to_atom:
            return True
        else:
            return False

    def __str__(self):
        return '<Bond from="{from_atom}" to="{to_atom}"/>'.format(**self.__dict__)

    __repr__ = __str__


class _Atom(object):
    """
        Private class representing an atom that is part of a single residue
    """

    def __init__(self, resname, name, number=None):
        self.resname = resname
        self.name = name

        #  If number not supplied, guess by looking for a number at the end of the atom name
        # numerical component in name. NOT equivalent to index in Residue block
        if number is None:
            self.number = int(re.findall(r"\d+", name)[-1])
        else:
            self.number = number
        self.type = "{}-{}".format(resname,name)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def _same_res(self, other):
        """Compare resnames"""
        return self.resname == other.resname

    def __int__(self):
        """Convert atom to integer."""
        return int(self.number) - 1

    def __gt__(self, other):
        if not self._same_res(other):
            raise ValueError("Cannot compare atoms between residues.")
        if self.number > other.number:
            return True
        else:
            return False

    def __lt__(self, other):
        if not self._same_res(other):
            raise ValueError("Cannot compare atoms between residues.")
        if self.number < other.number:
            return True
        else:
            return False

    def __le__(self, other):
        if not self._same_res(other):
            raise ValueError("Cannot compare atoms between residues.")
        if self.number <= other.number:
            return True
        else:
            return False

    def __ge__(self, other):
        if not self._same_res(other):
            raise ValueError("Cannot compare atoms between residues.")
        if self.number >= other.number:
            return True
        else:
            return False

    def __str__(self):
        return '<Atom name="{name}" type="{type}"/>'.format(**self.__dict__)

    def __repr__ (self):
        self.__str__() + str(self.number)


class _AtomType(object):
    """
        Private class representing an atomtype
    """
    def __init__(self, resname, atomname, aclass, element, mass):
        self.name = "{}-{}".format(resname, atomname)
        self.aclass = aclass
        self.element = element
        self.mass = float(mass)
        self.number = int(re.findall(r"\d+", atomname)[-1])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return '<Type name="{name}" class="{aclass}" element="{element}" mass="{mass}"/>'.format(**self.__dict__)

    __repr__ = __str__

    def nonbonded_placeholder(self):
        """
        Returns
        -------
        str - Placeholder for non-bonded block.
        """
        return '<Atom type="{name}" charge="0.0" sigma="1.0" epsilon="1.0"/>'.format(**self.__dict__)


class _NonbondedForce(object):
    """
    Private class representing the parameters for a nonbonded atom type
    """
    def __init__(self, resname, state, atomname, epsilon, sigma, charge):
        self.resname = resname
        self.state = state
        self.atomname = atomname
        self.epsilon = epsilon
        self.sigma = sigma
        self.charge = charge
        self.type = self.resname + "-" + self.atomname

    def __eq__(self, other):
        if not self.type == other.type:
            raise ValueError("Can only compare identical-atom types.")

        return self.sigma == other.sigma and self.epsilon == other.epsilon

    def __str__(self):
        return '<Atom type="{type}" charge="{charge}" sigma="{sigma}" epsilon="{epsilon}"/>'.format(**self.__dict__)

    __repr__ = __str__


class _Isomer(object):
    def __init__(self, index, population, charge):
        self.index = index
        self.population = population
        self.charge = charge
        self.nonbondedtypes = OrderedDict()

    def __str__(self):
        return '<IsomericState index="{index}" population="{population}" netcharge="{charge}"/>'.format(**self.__dict__)

    def __len__(self):
        return len(self.nonbondedtypes)

    __repr__ = __str__


class _TitratableForceFieldCompiler(object):
    """
    Compiles a
    """
    # TODO make sure we're not changing sigma/epsilon
    # format of atomtype name
    # Groups 1: resname, 2: isomer number, 3: atom letter, 4 atom number
    typename_format = re.compile(r"^(\w+)-(\d+)-(\D+)(\d+)$")

    def __init__(self, xmlfile, autoresolve=1):
        """
        Compliles the intermediate ffxml files into a constant-pH compatible ffxml file.

        Parameters
        ----------
        xmlfile - str
            Path of the intermediate xml file.
        autoresolve - int, optional
            Automatically resolve any type conflicts by picking the entry numbered. Unpredictable.
            Choose 0 unless you know exactly what you are doing.
            0 leads to an interactive resolution of conflicts (recommended!).
        """
        self._atoms = OrderedDict()
        self._bonds = list()
        self._atom_types = list()
        self._isostates = OrderedDict()
        xmlparser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        self._input_tree = etree.parse(xmlfile, parser=xmlparser)
        self._make_output_tree(autoresolve)

    def _make_output_tree(self, autoresolve):
        """Store all contents of a compiled ffxml file of all isomers, and add dummies for all missing hydrogens.
        """
        self._resname = None
        # Create a registry of atom names in self._atoms and atom types in self._atom_types
        self._complete_atom_registry(autoresolve=autoresolve)
        self._output_tree = _make_cph_xml()

        # Add missing atoms to shift indices of atoms in bonds correctly, then register bonds
        self._complete_bond_registry()

        # Register the isomeric states (without nonbonded parameters)
        self._complete_state_registry()

        # Add nonbonded terms for each state (excludes placeholders for the ForceField block)
        self._complete_nonbonded_registry(autoresolve=autoresolve)

        self._sort_bonds()
        self._sort_atypes()

        # Add atoms and bonds to the output
        for residue in self._output_tree.xpath('/ForceField/Residues/Residue'):
            residue.attrib["name"] = self._resname

            for atom in self._atoms.values():
                residue.append(etree.fromstring(str(atom)))

            for bond in self._bonds:
                residue.append(etree.fromstring(str(bond)))

        # Add atomtypes to the output
        for atypes in self._output_tree.xpath('/ForceField/AtomTypes'):
            for atype in self._atom_types:
                atypes.append(etree.fromstring(str(atype)))

        # Copy bonded definitions from input directly to output
        for hbfblock in self._output_tree.xpath('/ForceField/HarmonicBondForce'):
            for harmonicbondforce in self._input_tree.xpath('/TitratableResidue/ForceField/HarmonicBondForce/Bond'):
                hbfblock.append(harmonicbondforce)

        for hafblock in self._output_tree.xpath('/ForceField/HarmonicAngleForce'):
            for harmonicangleforce in self._input_tree.xpath('/TitratableResidue/ForceField/HarmonicAngleForce/Angle'):
                hafblock.append(harmonicangleforce)

        for pdtblock in self._output_tree.xpath('/ForceField/PeriodicTorsionForce'):
            for propertorsionforce in self._input_tree.xpath('/TitratableResidue/ForceField/PeriodicTorsionForce/Proper'):
                pdtblock.append(propertorsionforce)
            for impropertorsionforce in self._input_tree.xpath('/TitratableResidue/ForceField/PeriodicTorsionForce/Improper'):
                pdtblock.append(impropertorsionforce)

        # Fill in nonbonded atomtypes (placeholders)
        for forcefieldblock in self._output_tree.xpath('/ForceField'):
            for nonbondedblock in self._input_tree.xpath('/TitratableResidue/ForceField/NonbondedForce'):

                # Remove all old nonbonded terms
                for anonbond in nonbondedblock.xpath('/TitratableResidue/ForceField/NonbondedForce/Atom'):
                    anonbond.getparent().remove(anonbond)
                nonbondedblock.append(etree.Comment("Placeholder values. Each state has its own block in IsomerData."))

                # Make placeholder entries in the nonbonded block for each atom type.
                for atype in self._atom_types:
                    nonbondedblock.append((etree.fromstring(atype.nonbonded_placeholder())))
                forcefieldblock.append(nonbondedblock)

        for isodata in self._output_tree.xpath('/ForceField/Residues/Residue/IsomerData'):
            for isoindex, isostate in self._isostates.items():
                statexml = etree.fromstring(str(isostate))
                for nonbondtype in isostate.nonbondedtypes.values():
                    statexml.append(etree.fromstring(str(nonbondtype)))
                isodata.append(statexml)

        # TODO ADD GB solvent parameters

    def _complete_bond_registry(self):
        for residue in self._input_tree.xpath('/TitratableResidue/ForceField/Residues/Residue'):
            per_state_index = OrderedDict()
            for aix, atom in enumerate(residue.xpath('Atom')):

                per_state_index[aix] = atom.attrib["name"]

            for bond in residue.xpath('Bond'):
                atomnames = list(self._atoms.keys())
                from_atom = per_state_index[int(bond.attrib["from"])]
                to_atom = per_state_index[int(bond.attrib["to"])]

                self._bonds.append(_Bond(atomnames.index(from_atom), atomnames.index(to_atom)))

        self._unique_bonds()

    def write(self, filename=None, moleculename="UNK"):
        objectify.deannotate(self._output_tree)
        etree.cleanup_namespaces(self._output_tree)
        xmlstring = etree.tostring(self._output_tree, encoding="utf-8", pretty_print=True, xml_declaration=False)
        xmlstring = xmlstring.decode("utf-8")
        xmlstring = xmlstring.replace("non-", "{}-".format(moleculename))
        xmlstring = xmlstring.replace('name="non"', 'name="{}"'.format(moleculename))

        if filename is not None:
            with open(filename, 'w') as fstream:
                fstream.write(xmlstring)

        return xmlstring

    def _complete_atom_registry(self, autoresolve):
        """
        Registers unique atom names. Store in self._atomnames from atom type list.
        """
        for atype in self._input_tree.xpath('/TitratableResidue/ForceField/AtomTypes/Type'):
            matched_names = _TitratableForceFieldCompiler.typename_format.search(atype.attrib['name'])
            if self._resname is None:
                self._resname = matched_names.group(1)
            atomname = matched_names.group(3, 4)
            atomname = ''.join(atomname)
            if atomname not in self._atoms:
                self._atoms[atomname] = _Atom(self._resname, atomname)
            self._atom_types.append(_AtomType(self._resname, atomname, atype.attrib['class'], atype.attrib['element'], atype.attrib['mass']))

        self._unique_atom_types(autoresolve=autoresolve)
        self._sort_atoms()

    def _complete_state_registry(self):
        for isostate in self._input_tree.xpath('/TitratableResidue/IsomerData/IsomericState'):
            isomer = _Isomer(int(isostate.attrib["index"]), isostate.attrib["population"], isostate.attrib["netcharge"])
            self._isostates[int(isostate.attrib["index"])] = isomer

    def _complete_nonbonded_registry(self, autoresolve):
        """
        Register the non_bonded interactions for all states.
        """
        for isoindex, isostate in self._isostates.items():
            unmatched_atoms = list(self._atoms.keys())
            for nonbondatom in self._input_tree.xpath("/TitratableResidue/ForceField/NonbondedForce/Atom"):
                matched_names = _TitratableForceFieldCompiler.typename_format.search(nonbondatom.attrib['type'])
                atomname = matched_names.group(3, 4)
                atomname = ''.join(atomname)
                stateidx_nonbond = int(matched_names.group(2)) + 1
                if stateidx_nonbond == isoindex:
                    unmatched_atoms.remove(atomname)
                    self._isostates[isoindex].nonbondedtypes[atomname] = _NonbondedForce(self._resname, isoindex, atomname, nonbondatom.attrib['epsilon'], nonbondatom.attrib['sigma'], nonbondatom.attrib['charge'])

            for unmatched in unmatched_atoms:
                self._isostates[isoindex].nonbondedtypes[unmatched] = _NonbondedForce(self._resname, isoindex,
                                                                             unmatched, "0.0",
                                                                             "0.0",
                                                                             "0.0")

        for atom in self._atoms.values():
            self._resolve_lj_params(atom, autoresolve=autoresolve)

    def _resolve_lj_params(self, atom, autoresolve):
        typelist = OrderedDict()
        for isoindex, isostate in self._isostates.items():
            nbtype = isostate.nonbondedtypes[atom.name]
            # Assume zero values for all are newly added dummy atoms, and should not be suggested as types.
            if (float(nbtype.epsilon) > 0.0 and float(nbtype.sigma) > 0.0) or abs(float(nbtype.charge)) > 0.0:
               typelist[isoindex] = nbtype

        # Select an arbitrary type to compare all to
        randomindex = random.choice(list(typelist.keys()))
        first_state_type = typelist[randomindex]
        # Check if all remaining types have the same LJ parameters
        if all(atype == first_state_type for atype in typelist.values()):
            logger.debug("All types shared for {}".format(atom))
            # Still need to standardize the dummies.
            self._standardize_lj_params(atom, randomindex)
        elif autoresolve:
            if autoresolve not in typelist.keys():
                raise UserWarning("Nonbonded type reference provided a dummy atom, this is probably a bad idea.")
            for key, value in typelist.items():
                logger.debug("{}:{}".format(key, value))
            self._standardize_lj_params(atom, autoresolve)
        else:
            self._manual_lj_params(atom, typelist)

    def _standardize_lj_params(self, atom, reference_state_index):
        """
        Set the epsilon and sigma for the atom to that of the reference state, for all other states.
        """
        epsilon = self._isostates[reference_state_index].nonbondedtypes[atom.name].epsilon
        sigma = self._isostates[reference_state_index].nonbondedtypes[atom.name].sigma
        logger.info("Setting epsilon/sigma for {} to {} / {}".format(atom.name, epsilon, sigma))
        for isoindex in self._isostates.keys():
            self._isostates[isoindex].nonbondedtypes[atom.name].epsilon = epsilon
            self._isostates[isoindex].nonbondedtypes[atom.name].sigma = sigma

    def _manual_lj_params(self, atom, typelist):
        print("Atom types for {}".format(atom.name))
        for isoindex, atype in typelist.items():
            print("{}), {}".format(isoindex, atype))
        choice = -1
        while choice not in typelist.keys():
            print("Please pick a number from the list:")
            choice = int(input("Which sigma/epsilon would you like to keep?").strip())

        self._standardize_lj_params(atom, choice)

    def _unique_bonds(self):
        """Ensure only unique bonds are kept."""
        bonds = list()
        for bond in self._bonds:
            if bond not in bonds:
                bonds.append(bond)

        self._bonds = bonds

    def _unique_atom_types(self, autoresolve):
        """Ensure only unique atomtypes are kept. Provide conflict resolution."""
        atypes = list()
        for atype in self._atom_types:
            if atype not in atypes:
                atypes.append(atype)
        self._atom_types = atypes

        # Atomtypes might be different between isomers. We can only support one type at the moment
        while len(self._atom_types) != len(self._atoms):
            for atype1 in self._atom_types:
                conflicts = [atype1]
                for atype2 in self._atom_types:
                    if atype1 == atype2:
                        continue
                    elif atype1.name == atype2.name:
                        conflicts.append(atype2)

                    if len(conflicts) > 1:
                        self._resolve_type_conflict(conflicts, autoresolve=autoresolve)

    def _resolve_type_conflict(self, conflicts, autoresolve):
        """Keep all but one of the conflicting bonded atom types.

        Parameters
        ----------
        conflicts - list
            Conflicting atom types
        autoresolve - int
            one based index of the type to keep in each case.
            If not provided, interactive choice will be provided.

        Notes
        -----
        If the autoresolve index is out of bounds, will fallback to interactive mode.
        """
        if autoresolve:
            try:
                logger.info("Using atomtype {}".format(conflicts[autoresolve-1]))
                conflicts.remove(conflicts[autoresolve-1])
                logger.debug("Deleting atomtypes {}".format(conflicts))
                for conf in conflicts:
                    self._atom_types.remove(conf)
            except IndexError:
                logger.warning("Automatic type resolution failure (out of bounds.) Switching to manual.")
                self._manual_type_resolution(conflicts)
        else:
            self._manual_type_resolution(conflicts)

    def _manual_type_resolution(self, conflicts):
        while True:
            for c, conf in enumerate(conflicts, start=1):
                print("Type {}) : {}".format(c,conf))
            keep = int(input("Which type do you want to keep? (1/2)").strip())
            if keep not in range(1, len(conflicts)+1):
                print("Please pick a number from the list.")
            else:
                logger.info("Using atomtype {}".format(conflicts[keep-1]))
                conflicts.remove(conflicts[keep-1])
                logger.debug("Deleting atomtypes {}".format(conflicts))
                for conf in conflicts:
                    self._atom_types.remove(conf)
                break

    def _sort_atoms(self):
        self._atoms = OrderedDict(sorted(self._atoms.items(), key=lambda t: t[1]))  # Sort by atom number

    def _sort_bonds(self):
        self._bonds = sorted(self._bonds)  # Sort by first, then second atom in bond

    def _sort_atypes(self):
        self._atom_types = sorted(self._atom_types, key=lambda at: at.number)

    def _new_dummy_atom(self, atomname, typestr):
        newatom = etree.Element('Atom')
        newatom.set('name', atomname)
        newatom.set('type', typestr)
        return newatom

    def _new_dummy_nonbond(self, typestr):
        newnonbond = etree.Element('Atom')
        newnonbond.set('type', typestr)
        newnonbond.set('charge', "0.0")
        newnonbond.set('sigma', '0.0')
        newnonbond.set('epsilon', '0.0')
        return newnonbond

    def _new_dummy_type(self, typestr, mass=1.007947):
        newtype = etree.Element('Type')
        newtype.set('name', typestr)
        newtype.set('class', 'du')
        newtype.set('element', 'H')
        newtype.set('mass', str(mass))
        return newtype

    def _shift_bonds(self, start, residue):
        for bond in residue.findall('Bond'):
            b_from = int(bond.attrib['from'])
            b_to = int(bond.attrib['to'])
            if b_from > start:
                bond.attrib['from'] >= str(b_from + 1)
            if b_to > start:
                bond.attrib['to'] >= str(b_to + 1)


def _param_isomer(isomer, tmpdir, q):
    # Read temporary file containing net charge
    #TODO hardcoded using gasteiger charges for debugging speed
    omt.amber.run_antechamber('isomer_{}'.format(isomer),'{}/{}.mol2'.format(tmpdir,isomer), charge_method="bcc", net_charge=q)
    logging.info("Parametrized isomer{}".format(isomer))


def line_prepender(filename, line):
    #http://stackoverflow.com/a/5917395
    with open(filename, 'r+') as f:
        content = f.read() # current content of the file
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content) # add new line before the current content


def parametrize_ligand(inputmol2, outputffxml, max_antechambers=1, tmpdir=None, remove_temp_files=False, pH=7.4):
    """
    Parametrize a ligand for constant-pH simulation using Epik.

    Parameters
    ----------
    inputmol2 : str
        location of mol2 file with all possible atoms included.
    outputffxml : str
        location for output xml file containing all ligand states and their parameters

    Other Parameters
    ----------------
    pH : float
        The pH that Epik should use
    max_antechambers : int, optional (default : 1)
        Maximal number of concurrent antechamber processes.
    tmpdir : str, optional
        Temporary directory for storing intermediate files.
    remove_temp_files : bool, optional (default : True)
        Remove temporary files when done.

    Notes
    -----
    The supplied mol2 file needs to have ALL possible atoms included, with unique names.
    This could be non-physical, also don't worry about bond order.
    If you're not sure, better to overprotonate. Epik doesn't retain the input protonation if it's non-physical.
    
    Returns
    -------
    str : The absolute path of the outputfile
    """
    logger.setLevel(logging.DEBUG)
    logger.info("Running Epik to detect protomers and tautomers...")
    inputmol2 = os.path.abspath(inputmol2)
    outputffxml = os.path.abspath(outputffxml)
    oldwd = os.getcwd()
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    logger.debug("Running in {}".format(tmpdir))

    # Using very tolerant settings, since we can't predict probability from just ref states.
    omt.schrodinger.run_epik(inputmol2, "epik.mae", ph=pH, min_probability=0.00001, ph_tolerance=5.0)
    omt.schrodinger.run_structconvert("epik.mae", "epik.sdf")
    omt.schrodinger.run_structconvert("epik.mae", "epik.mol2")
    logger.info("Done with Epik run!")

    # Grab data from sdf file and make a file containing the charge and penalty
    logger.info("Processing Epik output...")
    epikxml = _make_xml_root("IsomerData")
    isomer_index = 0
    store = False
    charges = dict()
    for line in open('epik.sdf', 'r'):
        if store:
            value = line.strip()

            if store == "population":
                value = float(value)
                value /= 298.15 * 1.9872036e-3 # / RT in kcal/mol/K at 25 Celsius (Epik default)
                value = exp(-value)

            obj.set(store, str(value))

            # NOTE: relies on state penalty coming before charge
            if store == "netcharge":
                epikxml.append(obj)
                charges[isomer_index] = int(value)
                del(obj)

            store = ""

        elif "r_epik_State_Penalty" in line:
            # Next line contains epik state penalty
            store = "population"
            isomer_index += 1
            obj = objectify.Element("IsomericState")
            obj.set("index", str(isomer_index))

        elif "i_epik_Tot_Q" in line:
            # Next line contains charge
            store = "netcharge"


    # remove lxml annotation
    objectify.deannotate(epikxml)
    etree.cleanup_namespaces(epikxml)
    epikxmlstring = etree.tostring(epikxml, pretty_print=True, xml_declaration=False)

    # Make new mol2 file for each isomer
    with open('epik.mol2', 'r') as molfile:
        isomer_index = 0
        out = None
        for line in molfile:
            if line.strip() =='@<TRIPOS>MOLECULE':
                if out:
                    out.close()
                isomer_index += 1
                out = open('{}.mol2'.format(isomer_index), 'w')
            out.write(line)
        out.close()
    logger.info("Done! Processed {} isomers.".format(isomer_index))
    logger.info("Calculating isomer parameters {} processes at a time... (This may take a while!)".format(max_antechambers))
    Parallel(n_jobs=max_antechambers)(delayed(_param_isomer)(i, tmpdir, charges[i]) for i in range(1, isomer_index + 1))
    logger.info("Done!")
    logger.info("Combining isomers into one XML file.")

    omt.utils.create_ffxml_file(glob("isomer*.mol2"), glob("isomer*.frcmod"), ffxml_filename="intermediate.xml")

    # Append epik information to the end of the file
    with open("intermediate.xml", mode='ab') as outfile:
        outfile.write(epikxmlstring)
        outfile.write(b"</TitratableResidue>")
    line_prepender("intermediate.xml", "<TitratableResidue>")

    _TitratableForceFieldCompiler("intermediate.xml").write(outputffxml)
    logger.info("Done, your result is located here: {}!".format(outputffxml))
    if remove_temp_files:
        shutil.rmtree(tmpdir)
    os.chdir(oldwd)
    return outputffxml
