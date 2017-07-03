"""
Integrators that can be used with Protons' NCMC implementation
"""
import numpy
import simtk.openmm as mm
import simtk.unit
import simtk.unit as units
from openmmtools.integrators import ThermostatedIntegrator

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA


class GBAOABIntegrator(ThermostatedIntegrator):
    """This is a reference implementation of the gBAOAB integrator. For simplicity and reliability
    this omits the variable splitting schemes found in openmmtools.integrators.LangevinIntegrator.
    
    It accumulates external protocol work performed in the `protocol_work` variable, in OpenMM standard units (kJ/mol).
    
    To reset the protocol work, set the global variable `first_step` to 0. It will then be reset on the next iteration.
    Alternatively, 

     The different steps of the integrator are split up as followed(V(n*R)O(n*R)V), where the substeps are defined as 
     described by Leimkuhler and Matthews

        - R: Linear "drift" / Constrained "drift"
            Deterministic update of *positions*, using current velocities
            x <- x + v dt
        - V: Linear "kick" / Constrained "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass
        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal

    References
    ----------
    [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic and stochastic numerical methods, Chapter 7


    Main contributors (alphabetical order)
    --------------------------------------
    Josh Fass
    Patrick Grinaway    
    Gregory Ross
    Bas Rustenburg

    """

    def __init__(self, number_R_steps=1, temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=1.0 / simtk.unit.picoseconds, timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8):
        """
        Create a gBAOAB integrator using V (R * number_R_steps) * O (R * number_R_steps) V splitting.

        Parameters
        ----------
        number_of_R_steps : int, default: 1
            the number of R operations/2. (Total number of R operations is 2 * number of R steps) 
        temperature : simtk.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           The collision rate.
        timestep : simtk.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           The integration timestep.
        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver          

        """
        # Initialize constants.
        kT = kB * temperature
        gamma = collision_rate

        # Create a new custom integrator.
        super(GBAOABIntegrator, self).__init__(temperature, timestep)

        self.addPerDofVariable("sigma", 0)
        # Velocity mixing parameter: current velocity component
        self.addGlobalVariable("a", numpy.exp(-gamma * timestep))

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", numpy.sqrt(1 - numpy.exp(- 2 * gamma * timestep)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # The protocol work, calculated before the start of the next propagation step
        self.addGlobalVariable("protocol_work", 0)

        # The energy at the start of the next propagation step, assumes perturbation happened in between propagations
        self.addGlobalVariable("perturbed_pe", 0)

        # The energy after the propagation step.
        self.addGlobalVariable("unperturbed_pe", 0)

        # Binary toggle to indicate first step
        self.addGlobalVariable("first_step", 0)

        # Begin of integration procedure

        # Calculate the perturbation
        self.addComputeGlobal("perturbed_pe", "energy")

        # Assumes no perturbation is done before doing the initial MD step.
        self.beginIfBlock("first_step < 1")
        self.addComputeGlobal("first_step", "1")
        self.addComputeGlobal("unperturbed_pe", "energy")
        self.addComputeGlobal("protocol_work", "0.0")
        self.endBlock()
        # the protocol work is incremented
        self.addComputeGlobal("protocol_work", "protocol_work + (perturbed_pe - unperturbed_pe)")

        # Update temperature/barostat dependent state
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        # V step
        self.addComputePerDof("v", "v + (dt / 2) * f / m")
        self.addConstrainVelocities()

        # R step(s)
        for i in range(number_R_steps):
            self.addComputePerDof("x", "x + ((dt / {}) * v)".format(number_R_steps * 2))
            self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
            self.addConstrainPositions()  # x is now constrained
            self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(number_R_steps * 2))
            self.addConstrainVelocities()

        # O step
        self.addComputePerDof("v", "(a * v) + (b * sigma * gaussian)")
        self.addConstrainVelocities()

        # R step(s)
        for i in range(number_R_steps):
            self.addComputePerDof("x", "x + ((dt / {}) * v)".format(number_R_steps * 2))
            self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
            self.addConstrainPositions()  # x is now constrained
            self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(number_R_steps * 2))
            self.addConstrainVelocities()

        # V step
        self.addComputePerDof("v", "v + (dt / 2) * f / m")
        self.addConstrainVelocities()

        # Calculate the potential energy after propagation step
        self.addComputeGlobal("unperturbed_pe", "energy")

    def reset_protocol_work(self):
        """Reset protocol work tracking.
        
        This is a shortcut to resetting the accumulation of protocol work.
        """
        self.setGlobalVariableByName("first_step", 0)
        self.setGlobalVariableByName('protocol_work', 0)


class GHMCIntegrator(mm.CustomIntegrator):
    """

    This generalized hybrid Monte Carlo (GHMC) integrator is a modification of the GHMC integrator found
    here https://github.com/choderalab/openmmtools. Multiple steps can be taken per integrator.step() in order to save
    the potential energy before the steps were made.

    Authors: John Chodera and Gregory Ross
    """

    def __init__(self, temperature=298.0 * simtk.unit.kelvin, collision_rate=91.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds, nsteps=1):
        """
        Create a generalized hybrid Monte Carlo (GHMC) integrator.

        Parameters
        ----------
        temperature : simtk.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           The collision rate.
        timestep : simtk.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           The integration timestep.
        nsteps : int
           The number of steps to take per integrator.step()

        Notes
        -----
        It is equivalent to a Langevin integrator in the velocity Verlet discretization with a
        Metrpolization step to ensure sampling from the appropriate distribution.

        An additional global variable 'potential_initial' records the potential energy before 'nsteps' have been taken.

        Example
        -------

        Create a GHMC integrator.

        >>> temperature = 298.0 * simtk.unit.kelvin
        >>> collision_rate = 91.0 / simtk.unit.picoseconds
        >>> timestep = 1.0 * simtk.unit.femtoseconds
        >>> integrator = GHMCIntegrator(temperature, collision_rate, timestep)

        References
        ----------
        Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
        http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472

        """
        # Initialize constants.
        kT = kB * temperature
        gamma = collision_rate

        # Create a new custom integrator.
        super(GHMCIntegrator, self).__init__(timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("kT", kT)  # thermal energy
        self.addGlobalVariable("b", numpy.exp(-gamma * timestep))  # velocity mixing parameter
        self.addPerDofVariable("sigma", 0)  # velocity standard deviation
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("vold", 0)  # old velocities
        self.addPerDofVariable("xold", 0)  # old positions
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("potential_initial", 0)  # initial potential energy
        self.addGlobalVariable("potential_old", 0)  # old potential energy
        self.addGlobalVariable("potential_new", 0)  # new potential energy
        self.addGlobalVariable("protocol_work", 0)
        self.addGlobalVariable("accept", 0)  # accept or reject
        self.addGlobalVariable("naccept", 0)  # number accepted
        self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials
        self.addPerDofVariable("x1", 0)  # position before application of constraints
        self.addGlobalVariable("step", 0)  # variable to keep track of number of propagation steps
        self.addGlobalVariable("nsteps", nsteps)  # The number of iterations per integrator.step(1).
        #
        # Initialization.
        #
        self.beginIfBlock("ntrials = 0")
        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addComputeGlobal("protocol_work", "0.0")
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.addComputeGlobal("potential_new", "energy")
        self.endBlock()

        #
        # Allow context updating here.
        #
        self.addUpdateContextState()

        self.addComputeGlobal("potential_initial", "energy")
        self.addComputeGlobal("step", "0")
        self.addComputeGlobal("protocol_work", "protocol_work + (potential_initial - potential_new)")
        if True:
            self.beginWhileBlock("step < nsteps")
            #
            # Velocity randomization
            #
            self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
            self.addConstrainVelocities()

            # Compute initial total energy
            self.addComputeSum("ke", "0.5*m*v*v")
            self.addComputeGlobal("potential_old", "energy")
            self.addComputeGlobal("Eold", "ke + potential_old")
            self.addComputePerDof("xold", "x")
            self.addComputePerDof("vold", "v")
            # Velocity Verlet step
            self.addComputePerDof("v", "v + 0.5*dt*f/m")
            self.addComputePerDof("x", "x + v*dt")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
            self.addConstrainVelocities()
            # Compute final total energy
            self.addComputeSum("ke", "0.5*m*v*v")
            self.addComputeGlobal("potential_new", "energy")
            self.addComputeGlobal("Enew", "ke + potential_new")
            # Accept/reject, ensuring rejection if energy is NaN
            self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
            self.beginIfBlock("accept != 1")
            self.addComputePerDof("x", "xold")
            self.addComputePerDof("v", "-vold")
            self.addComputeGlobal("potential_new", "potential_old")
            self.endBlock()
            #
            # Velocity randomization
            #
            self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
            self.addConstrainVelocities()
            #
            # Accumulate statistics.
            #
            self.addComputeGlobal("naccept", "naccept + accept")
            self.addComputeGlobal("ntrials", "ntrials + 1")

            self.addComputeGlobal("step", "step+1")
            self.endBlock()
