
import numpy as np


class ScipyConstraints(object):

    def __init__(self):
        self.constraints = []

    def add_gte(self, larger_param, smaller_param):

        def lagrangian(pv, pi):
            return pv[pi[larger_param]] - pv[pi[smaller_param]]

        def jacobian(_, pi):
            j = np.zeros(len(pi))
            j[pi[larger_param]] = 1.0
            j[pi[smaller_param]] = -1.0
            return j

        self.constraints.append({
            'type': 'ineq',
            'fun': lagrangian,
            'jac': jacobian
        })
        return self

    def add_gtez(self, param):

        def lagrangian(pv, pi):
            return pv[pi[param]]

        def jacobian(_, pi):
            j = np.zeros(len(pi))
            j[pi[param]] = 1.0
            return j

        self.constraints.append({
            'type': 'ineq',
            'fun': lagrangian,
            'jac': jacobian
        })
        return self

    def add_gtev(self, param, value):

        def lagrangian(pv, pi):
            return pv[pi[param]] - value

        def jacobian(_, pi):
            j = np.zeros(len(pi))
            j[pi[param]] = 1.0
            return j

        self.constraints.append({
            'type': 'ineq',
            'fun': lagrangian,
            'jac': jacobian
        })
        return self

    def add_ltez(self, param):

        def lagrangian(pv, pi):
            return -1 * pv[pi[param]]

        def jacobian(_, pi):
            j = np.zeros(len(pi))
            j[pi[param]] = -1.0
            return j

        self.constraints.append({
            'type': 'ineq',
            'fun': lagrangian,
            'jac': jacobian
        })
        return self

    def get_constraints(self):
        # Return copied constraint definitions
        return [dict(c) for c in self.constraints]

    def merge(self, constraints):
        """ Merge constraint set with another
        :param constraints: Other constraint set to add to this one
        :return: New constraint set object
        """
        res = ScipyConstraints()
        res.constraints = self.get_constraints() + constraints.get_constraints()
        return res
