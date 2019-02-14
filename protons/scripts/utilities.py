from openmmtools.integrators import ExternalPerturbationLangevinIntegrator
from simtk import openmm as mm, unit
from lxml import etree, objectify
from protons import app
from protons.app import log
import numpy as np
from io import StringIO

xmls = mm.XmlSerializer
import datetime


class TimeOutError(RuntimeError):
    """This error is raised when an operation is taking longer than expected."""

    pass


def timeout_handler(signum, frame):
    """Handle a timeout."""
    log.warn("Script is running out of time. Attempting to exit cleanly.")
    raise TimeOutError("Running out of time, shutting down!")


def serialize_state(context) -> str:
    """
    Serialize the simulation state to xml.
    """
    statexml = xmls.serialize(
        context.getState(
            getPositions=True,
            getVelocities=True,
            getParameters=True,
            enforcePeriodicBox=True,
        )
    )
    return statexml


def serialize_system(system) -> str:
    """Serialize openmm system"""
    return xmls.serialize(system)


def deserialize_openmm_element(element: etree.Element):
    """Deserialize an lxml Element containing a serialized openmm object"""
    return mm.XmlSerializer.deserialize(etree.tostring(element, encoding="unicode"))


def serialize_integrator(integrator) -> str:
    """Serialize openmm integrator"""
    return xmls.serialize(integrator)


def serialize_drive(drive) -> str:
    """
    Serialize the drive residues and calibration state to xml.
    """
    drivexml = drive.state_to_xml()
    return drivexml


def num_to_int_str(num: float) -> str:
    """Takes a float and converts to int, and then returns a string version."""
    return str(int(round(num)))


def store_saltswap_state(salinator):
    """Store the saltswap state as xml so an identical salinator can be instantiated."""
    swapper = salinator.swapper
    root = etree.fromstring("<Saltswap><StateVector/></Saltswap>")
    root.set("salt_concentration_molar", str(salinator.salt_concentration / unit.molar))
    vecstring = " ".join(map(num_to_int_str, swapper.stateVector))
    root.xpath("StateVector")[0].text = vecstring
    return etree.tostring(root)


def store_topology(topology_file_string: str, fileformat="pdbx"):
    """Store a file (containing topology, such as pdbx) in a topology XML block."""
    root = etree.fromstring(f'<TopologyFile format="{fileformat}"/>')
    root.text = topology_file_string
    return root


def xml_to_topology(topology_elem: etree.Element) -> app.Topology:
    """From a TopologyFile xml element, create a topology"""
    text = topology_elem.text
    fileio = StringIO(text)

    topo_format = topology_elem.attrib["format"].lower()
    if topo_format == "pdbx":
        loaded = app.PDBxFile(fileio)
    elif topo_format == "pdb":
        loaded = app.PDBFile(fileio)
    else:
        raise ValueError(f"Unsupported topology format: '{topo_format}'.")

    return loaded.getTopology()


def create_calibration_checkpoint_file(
    filename: str,
    drive: app.ForceFieldProtonDrive,
    context: mm.Context,
    system: mm.System,
    integrator: mm.CustomIntegrator,
    topology_string: str,
    salinator=None,
) -> None:
    """Write out a checkpoint file for calibration-v1 example scripts."""

    date = str(datetime.datetime.now())
    runtype = "calibration-v1"  # hash version of script in future?

    drivexml = serialize_drive(drive)
    integratorxml = serialize_integrator(integrator)
    systemxml = serialize_system(system)
    state_xml = serialize_state(context)
    topology_xml = store_topology(topology_string)

    tree = etree.fromstring(
        f"""<Checkpoint runtype="{runtype}" date="{date}"></Checkpoint>"""
    )
    tree.append(etree.fromstring(drivexml))
    tree.append(etree.fromstring(integratorxml))
    tree.append(etree.fromstring(systemxml))
    tree.append(etree.fromstring(state_xml))
    tree.append(topology_xml)
    if salinator is not None:
        tree.append(etree.fromstring(store_saltswap_state(salinator)))

    with open(filename, "wb") as ofile:
        ofile.write(etree.tostring(tree))


def deserialize_state_vector(saltswap_tree: etree.Element, swapper) -> None:
    """Set the saltswap state vector from one stored in xml (inplace)."""
    vectsring = saltswap_tree.xpath("StateVector")[0].text
    swapper.stateVector = np.fromstring(vectsring, dtype=int, sep=" ")


class ExternalGBAOABIntegrator(ExternalPerturbationLangevinIntegrator):
    """
    Implementation of the gBAOAB integrator which tracks external protocol work.

    Parameters
    ----------
        number_R: int, default: 1
            The number of sequential R steps.  For instance V R R O R R V has number_R = 2
        temperature : simtk.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
        The temperature.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, default: 1.0/unit.picoseconds
        The collision rate.
        timestep : simtk.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
        The integration timestep.


    """

    def __init__(
        self,
        number_R_steps=1,
        temperature=298.0 * unit.kelvin,
        collision_rate=1.0 / unit.picoseconds,
        timestep=1.0 * unit.femtoseconds,
        constraint_tolerance=1e-7,
    ):
        Rstep = " R" * number_R_steps

        super(ExternalGBAOABIntegrator, self).__init__(
            splitting="V{0} O{0} V".format(Rstep),
            temperature=temperature,
            collision_rate=collision_rate,
            timestep=timestep,
            constraint_tolerance=constraint_tolerance,
            measure_shadow_work=False,
            measure_heat=False,
        )
