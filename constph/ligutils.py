from __future__ import print_function

from glob import glob
import openmoltools as omt
from joblib import Parallel, delayed
from lxml import etree
import re, os, sys, logging, tempfile, shutil


class _Bond(object):
    """
    Private class representing a bond between two atoms. Supports comparisons.
    """
    def __init__(self,atom1, atom2):
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


class MultiIsomerResidue(object):
    dummy_nonb='<Atom type="" charge="0.0" sigma="0.0" epsilon="0.0"/>'
    name_format=re.compile(r"^(\w+)-(\d+)-(\D+)(\d+)$")

    def __init__(self, xmlfile):
        self._atoms = set()
        self._bonds = list()
        xmlparser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        self._tree = etree.parse(xmlfile, parser=xmlparser)
        self._parse_ffxml(xmlfile)
        pass

    def _parse_ffxml(self, xmlfile):
        """Store all contents of a compiled ffxml file of all isomers.

        Parameters
        ----------
        xmlfile - str,
            An ffxml file containing isomers parametrized with GAFF.
        """
        # format of atomtype names resname(single word)-isomer(number)-atomname(non-number)atomnumber(number)

        # Create a registry of atom names in self._atoms
        self._register_atoms()

        for residue in self._tree.xpath('/ForceField/Residues/Residue'):
            lastnum=0
            for atom in residue.xpath('Atom'):
                atomnum = int(MultiIsomerResidue.name_format.search(atom.attrib['type']).group(4))
                
                missing_atoms = atomnum - lastnum
                # Count backwards
                for relative_index in range(missing_atoms-1, 0, -1):
                    
                    # Missing number is the last atom found - the missing index
                    missingnum = atomnum - relative_index
                    self._new_dummy(missingnum, residue)

                lastnum = atomnum

            # Add missing atoms at end
            for atomnum in range(lastnum+1, len(self._atoms)+1):
                # One based index of number missing from end of residue
                self._new_dummy(atomnum, residue)

            # Create a registry of bonds in self._bonds
            self._register_bonds()

        # Remove all bonds from current tree
        for residue in self._tree.xpath('/ForceField/Residues/Residue'):
            for bond in residue.findall('Bond'):
                bond.getparent().remove(bond)

        # Reintroduce all necessary bonds from stored list
        self._sort_bonds()
        for residue in self._tree.xpath('/ForceField/Residues/Residue'):
            for bond in self._bonds:
                residue.append(etree.fromstring(str(bond)))

    def write(self, filename):

        self._tree.write(filename, pretty_print=True)

    def _new_dummy(self, atom_index, residue):
        """

        Parameters
        ----------
        atom_index - int
            One-based index of the atom
        residue - etree.Element
            Residue to add atom to.

        Returns
        -------

        """
        atomname = 'X{}'.format(atom_index)
        typestr = ('{}-' + atomname).format(residue.attrib['name'])
        newtype = self._new_dummy_type(typestr)
        newatom = self._new_dummy_atom(atomname, typestr)
        newnonbond = self._new_dummy_nonbond(typestr)
        self._tree.xpath('/ForceField/AtomTypes')[0].append(newtype)
        self._tree.xpath('/ForceField/NonbondedForce')[0].append(newnonbond)
        residue.insert(atom_index - 1, newatom)
        self._shift_bonds(atom_index, residue)

    def _register_atoms(self):
        """
        Registers unique atom names. Store in self._atomnames from atom type list.
        """
        for atype in self._tree.xpath('/ForceField/AtomTypes/Type'):
            atomname = MultiIsomerResidue.name_format.search(atype.attrib['name']).group(3, 4)
            self._atoms.add(''.join(atomname))

    def _register_bonds(self):
        """
        Register the unique bonds. Store in self._bonds.

        Notes
        -----
        Only run this after all atom indices have been finalized.

        """
        for bond in self._tree.xpath('/ForceField/Residues/Residue/Bond'):
            fr = bond.attrib['from']
            to = bond.attrib['to']
            # etree.tostring(bond)
            self._bonds.append(_Bond(fr, to))

        self._unique_bonds()

    def _unique_bonds(self):
        bonds = list()

        for bond in self._bonds:
            if not bond in bonds:
                bonds.append(bond)

        self._bonds = bonds

    def _sort_bonds(self):
        self._bonds = sorted(self._bonds)

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
                bond.attrib['from'] = str(b_from + 1)
            if b_to > start:
                bond.attrib['to'] = str(b_to + 1)


def _param_isomer(isomer, tmpdir):
    os.chdir(tempfile.mkdtemp())
    q = int(open('{}/{}.q'.format(tmpdir,isomer)).read())
    omt.amber.run_antechamber('isomer_{}'.format(isomer),'{}/{}.mol2'.format(tmpdir,isomer), charge_method="bcc", net_charge=q)
    logging.info("Parametrized isomer{}".format(isomer))


def parametrize_ligand(inputmol2, outputxml, max_antechambers=1, tmpdir=None, remove_temp_files=True, pH=7.4):
    """
    Parametrize a ligand for constant-pH simulation.


    Parameters
    ----------
    inputmol2 : str
        location of mol2 file with all possible atoms included.
    outputxml : str
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

    logger = logging.getLogger()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info("Running Epik to detect protomers and tautomers...")
    inputmol2 = os.path.abspath(inputmol2)
    outputxml = os.path.abspath(outputxml)
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    # Using very tolerant settings, since we can't predict probability from just ref states.
    omt.schrodinger.run_epik(inputmol2, "epik.mae", ph=pH, min_probability=0.00001, ph_tolerance=5.0)
    omt.schrodinger.run_structconvert("epik.mae", "epik.sdf")
    omt.schrodinger.run_structconvert("epik.mae", "epik.mol2")
    logger.info("Done with Epik run!")

    # Grab data from sdf file and make a file containing the charge and penalty
    logger.info("Processing Epik output...")
    count = 0
    store = False
    for line in open('epik.sdf', 'r'):
        if store:
            out.write(line.strip())
            out.close()
            store = False
        elif "r_epik_State_Penalty" in line:
            store = True
            count += 1
            out = open('{}.pen'.format(count), 'w')
        elif "i_epik_Tot_Q" in line:
            store = True
            out = open('{}.q'.format(count), 'w')

    # Make new mol2 file for each isomer
    with open('epik.mol2', 'r') as molfile:
        count = 0
        out = None
        for line in molfile:
            if line.strip() =='@<TRIPOS>MOLECULE':
                if out:
                    out.close()
                count += 1
                out = open('{}.mol2'.format(count), 'w')
            out.write(line)
        out.close()
    logger.info("Done! Processed {} isomers.".format(count))
    logger.info("Calculating isomer parameters {} processes at a time... (This may take a while!)".format(max_antechambers))
    Parallel(n_jobs=max_antechambers)(delayed(_param_isomer)(i, tmpdir) for i in range(1, count + 1))
    logger.info("Done!")
    logger.info("Combining isomers into one XML file.")
    omt.utils.create_ffxml_file(glob("isomer*.mol2"), glob("isomer*.frcmod"), ffxml_filename=outputxml)
    logger.info("Done, your result is located here: {}!".format(outputxml))
    if remove_temp_files:
        shutil.rmtree(tmpdir)
    return outputxml
