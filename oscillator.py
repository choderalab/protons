import numpy as np
from scipy.misc import logsumexp
import random


class Oscillator(object):
    """Implementation of self-adjusted mixture sampling for calibrating titratable residues.
    """

    def __init__(self, x=0.0):
        self.states = np.asarray([0, 1])
        self.target_weights = np.asarray([1/2., 1/2.])
        self.x = x
        self.force_constant = 1
        self.j = random.choice(self.states)
        self.zeta = [0.0, 0.0]
        self.n_adaptations = 0

    def sample_configuration(self):
        sigma = 1.0 / np.sqrt(self.force_constant + self.j)
        self.x = sigma * np.random.normal() / np.pi

    def sample_state(self):
        """
        Sample a state using Gibbs sampling.

        Parameters
        ----------
        x : float
            Fixed configuration
        """
        logP_k = np.log(self.target_weights) - self.zeta - self._get_potential_energies(1, self.x)
        logP_k -= logsumexp(logP_k)
        P_k = np.exp(logP_k)
        self.j = np.random.choice(self.states, p=P_k)

    def adapt_weights(self, scheme):
        self.n_adaptations += 1
        beta = 1.0
        # zeta^{t-1}
        zeta = self.zeta

        if scheme in ['theorem1', 'eq9']:
            update = self._theorem1(self.target_weights)
        elif scheme in ['theorem2', 'eq12']:
            update = self._theorem2(self.target_weights, self.x, beta, zeta)
        else:
            raise ValueError("Unknown adaptation scheme!")

        # zeta^{t-1/2}
        zeta += update
        # zeta^{t} = zeta^{t-1/2} - zeta_1^{t-1/2}
        zeta_t = zeta - zeta[0]

        # Set reference energy based on new zeta
        for i, titr_state in enumerate(zeta_t):
            self.zeta = titr_state / -beta

    def _theorem1(self, target_weights):
        # [1/pi_1...1/pi_i]
        update = np.apply_along_axis(lambda x: 1/x, 0, target_weights)
        # delta(Lt)
        delta = np.zeros_like(update)
        delta[self.j] = 1
        update *= delta
        update /= self.n_adaptations  # t^{-1}
        return update

    def _theorem2(self, target_weights, x, beta, zeta):
        # target weights
        pi_j = self.target_weights
        # [1/pi_1...1/pi_i]
        update = np.apply_along_axis(lambda pi: 1/pi, 0, pi_j)
        ub_j = self._get_potential_energies(beta, x)
        # w_j(X;zeta)
        log_w_j = np.log(pi_j) - zeta - ub_j
        log_w_j -= logsumexp(log_w_j)
        w_j = np.exp(log_w_j)
        update *= w_j
        update /= self.n_adaptations  # t^{-1}
        return update

    def _get_potential_energies(self, beta, x):
        # beta * U(x)_j
        ub_j = np.empty(self.states.size)
        for j in range(self.states.size):
            ub_j[j] = beta * (j+self.force_constant) * x * x / 2

        return ub_j


#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    #
    # Test with an example from the Amber 11 distribution.
    #
    
    # Parameters.
    niterations = 5000 # number of dynamics/titration cycles to run
    nsteps = 500 # number of timesteps of dynamics per iteration

    system = Oscillator()
    sampled_states = []

    for iteration in range(niterations):

        for step in range(nsteps):
            system.sample_configuration()

        system.sample_state()
        system.adapt_weights('eq12')
        sampled_states.append(system.j)

    for j in system.states:
        print("state %d: %d"%(j, sampled_states.count(j)))
