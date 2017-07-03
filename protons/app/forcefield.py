# coding=utf-8
"""Augmented OpenMM.app.ForceField."""

from simtk.openmm.app import forcefield
from lxml import etree
import os


class ForceField(forcefield.ForceField):
    """A ForceField constructs OpenMM System objects based on a Topology. This version is augmented to
    know the locations of the protons ffxml files.

    Credit
    ------
    This class is a minor modification of the original ForceField class written by Peter Eastman.
    """

    def loadFile(self, file):
        """Load an XML file and add the definitions from it to this ForceField.

        Parameters
        ----------
        file : string or file
            An XML file containing force field definitions.  It may be either an
            absolute file path, a path relative to the current working
            directory, a path relative to this module's data subdirectory (for
            built in force fields), or an open file-like object with a read()
            method from which the forcefield XML data can be loaded.
        """
        try:
            # this handles either filenames or open file-like objects
            tree = etree.parse(file)

        except IOError:
            # first try protons xml files
            try:
                tree = etree.parse(os.path.join(os.path.dirname(__file__), 'data', file))
            # Try OpenMM default xml files.
            except IOError:
                tree = etree.parse(os.path.join(os.path.dirname(forcefield.__file__), 'data', file))
        except Exception as e:
            # Fail with an error message about which file could not be read.
            # TODO: Also handle case where fallback to 'data' directory encounters problems,
            # but this is much less worrisome because we control those files.
            msg = str(e) + '\n'
            if hasattr(file, 'name'):
                filename = file.name
            else:
                filename = str(file)
            msg += "ForceField.loadFile() encountered an error reading file '%s'\n" % filename
            raise Exception(msg)

        root = tree.getroot()

        # Load the atom types.

        if tree.getroot().find('AtomTypes') is not None:
            for type in tree.getroot().find('AtomTypes').findall('Type'):
                self.registerAtomType(type.attrib)

        # Load the residue templates.

        if tree.getroot().find('Residues') is not None:
            for residue in root.find('Residues').findall('Residue'):
                resName = residue.attrib['name']
                template = ForceField._TemplateData(resName)
                atomIndices = {}
                for atom in residue.findall('Atom'):
                    params = {}
                    for key in atom.attrib:
                        if key not in ('name', 'type'):
                            params[key] = forcefield._convertParameterToNumber(atom.attrib[key])
                    atomName = atom.attrib['name']
                    if atomName in atomIndices:
                        raise ValueError('Residue '+resName+' contains multiple atoms named '+atomName)
                    atomIndices[atomName] = len(template.atoms)
                    typeName = atom.attrib['type']
                    template.atoms.append(ForceField._TemplateAtomData(atomName, typeName, self._atomTypes[typeName].element, params))
                for site in residue.findall('VirtualSite'):
                    template.virtualSites.append(ForceField._VirtualSiteData(site, atomIndices))
                for bond in residue.findall('Bond'):
                    if 'atomName1' in bond.attrib:
                        template.addBondByName(bond.attrib['atomName1'], bond.attrib['atomName2'])
                    else:
                        template.addBond(int(bond.attrib['from']), int(bond.attrib['to']))
                for bond in residue.findall('ExternalBond'):
                    if 'atomName' in bond.attrib:
                        template.addExternalBondByName(bond.attrib['atomName'])
                    else:
                        template.addExternalBond(int(bond.attrib['from']))
                self.registerResidueTemplate(template)

        # Load force definitions

        for child in root:
            if child.tag in forcefield.parsers:
                forcefield.parsers[child.tag](child, self)

        # Load scripts

        for node in tree.getroot().findall('Script'):
            self.registerScript(node.text)