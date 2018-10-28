from simtk.openmm import app
from lxml import etree, objectify
import os


obc_block= '<GBSAOBCForce>\n  <UseAttributeFromResidue name="charge"/>\n</GBSAOBCForce>'



class OBCType:
    """A single OBC atom type, without charge."""

    def __init__(self, typename, radius, scale):
        """Add all parameters except charge."""

        self._typename = typename
        self._radius = radius
        self._scale = scale

    def to_xml(self):
        return '<Atom type="{_typename}" radius="{_radius:f}" scale="{_scale:f}"/>\n'.format(**self.__dict__)

    def __repr__(self):
        return self.to_xml()

    def __str__(self):
        return self.to_xml()

datadirs = app.forcefield._getDataDirectories()

# Find simtk/openmm/app/data
source = None
for directory in datadirs:
    head,tail = os.path.split(directory)
    head, tail2 = os.path.split(head)
    head, tail3 = os.path.split(head)

    if ["protons", "app", "data"] == [tail3, tail2, tail]:
        source = directory


# fail deadly
if source is None:
    raise Exception("Can't find the protons data directory.")


# from https://raw.githubusercontent.com/choderalab/gbff/27860c88fc5cfefea96f49558f7dd4bbfcd7edb0/parameters/gbsa-amber-mbondi2.parameters

types = {"H": (1.20 / 10.0, 0.85),
         "HN": (1.30 / 10.0, 0.85),
         "C": (1.70 / 10.0, 0.72),
         "N": (1.55 / 10.0, 0.79),
         "O": (1.50 / 10.0, 0.85),
         "F": (1.50 / 10.0, 0.88),
         "Si": (2.10 / 10.0, 0.80),
         "P": (1.85 / 10.0, 0.86),
         "S": (1.80 / 10.0, 0.96),
         "Cl": (1.70 / 10.0, 0.80),
         "Br": (1.50 / 10.0, 0.80),
         "I": (1.50 / 10.0, 0.80)}

for forcefieldfile in ["amber10-constph.xml", "gaff.xml", 'gaff2.xml']:
    root, ext = os.path.splitext(forcefieldfile)

    with open(forcefieldfile, 'r') as xmlio:
        tree = etree.fromstring(xmlio.read())

    obctypes = list()

    for atype in tree.xpath("/ForceField/AtomTypes/Type"):
        elem = atype.get("element")
        name = atype.get("class")

        if elem in types:
            if name.upper() == "HN":
                obctypes.append(OBCType(name, *types["HN"]))
            else:
                obctypes.append(OBCType(name, *types[elem]))

        else:
            print("No GB type for ", etree.tostring(atype).decode("UTF-8"))

    forceblock = etree.fromstring(obc_block)

    for otype in obctypes:
        forceblock.append(etree.fromstring(str(otype)))

    ff = etree.Element("ForceField")
    ff.append(forceblock)
    outtree= etree.ElementTree(ff)

    outxmlname = "{}-obc2-tmp.xml".format(root)
    with open(outxmlname, "wb") as outio:
        outtree.write(outio, pretty_print=True)

    # Ensure openmm can read the file without trouble, or clashes with the main forcefield
    app.ForceField(forcefieldfile, outxmlname)




