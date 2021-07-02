# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:26:41 2013

@author: wangronin
"""

import numpy as np
from numpy import (
    add,
    append,
    arange,
    argsort,
    array,
    atleast_2d,
    ceil,
    diag,
    dot,
    exp,
    eye,
    floor,
    inf,
    inner,
    isinf,
    isreal,
    linspace,
    log,
    mod,
    newaxis,
    ones,
    outer,
    power,
    r_,
    real,
    size,
    sqrt,
    triu,
    zeros,
)
from numpy.linalg import LinAlgError, cond, eigh, qr
from numpy.random import rand, randn, shuffle
from scipy.stats import chi

# import hello as h
from .boundary_handling import boundary_handling

# My fast routines...
abs = np.abs
max = np.max
min = np.min
sum = np.sum
norm = np.linalg.norm
any = np.any
all = np.all

# TODO: modularize this function, and possiblly fit it into a optimizer class
class cma_es(object):
    """

    My toy CMA-ES... with lots of variants of mutation operators...
    TODO: complete Python __doc__ of this function
    """

    def __init__(
        self,
        dim,
        init_wcm,
        fitnessfunc,
        opts,
        sampling_method=0,
        is_register=False,
        is_minimize=True,
        restart="IPOP",
    ):

        self.stop_dict = {}
        self.offspring = None
        self.sel = None
        self.z = None
        self.evalcount = 0
        self.eigeneval = 0
        self.fitnessfunc = fitnessfunc if is_minimize else lambda x: -fitnessfunc(x)
        self.fitness = None
        self.fitness_rank = None
        self.sampling_method = sampling_method
        self.is_minimize = is_minimize

        # TODO: implement BIPOP
        if restart == "IPOP":
            self.restart = restart
            self.restart_count = 0
            self.inc_popsize = 2
            self.restart_budget = opts["restart_budget"] if "restart_budget" in opts else int(20)

        # Initialize internal strategy parameters
        self.wcm = eval(init_wcm) if isinstance(init_wcm, str) else init_wcm
        self.lb = eval(opts["lb"]) if isinstance(opts["lb"], str) else opts["lb"]
        self.ub = eval(opts["ub"]) if isinstance(opts["ub"], str) else opts["ub"]
        self.lb = atleast_2d(self.lb)
        self.ub = atleast_2d(self.ub)

        if self.lb.shape[1] != 1:
            self.lb = self.lb.T
        if self.ub.shape[1] != 1:
            self.ub = self.ub.T

        self.eval_budget = (
            int(eval(opts["eval_budget"]))
            if isinstance(opts["eval_budget"], str)
            else int(opts["eval_budget"])
        )

        self.wcm = self.wcm.reshape(-1, 1)
        self.dim = dim
        self.sigma0 = opts["sigma_init"]
        self.sigma = self.sigma0
        self.f_target = opts["f_target"] if self.is_minimize else -opts["f_target"]

        # Strategy parameters: Selection
        self._lambda = opts["_lambda"] if "_lambda" in opts else int(4 + floor(3 * log(dim)))
        if isinstance(self._lambda, str):
            self._lambda = eval(self._lambda)
        _mu_prime = (self._lambda - 1) / 2.0
        self._mu = opts["_mu"] if "_mu" in opts else int(ceil(_mu_prime))
        if isinstance(self._mu, str):
            self._mu = eval(self._mu)

        # TODO : new weight setting weighted recombination
        self.weights = log(_mu_prime + 1.0) - log(arange(1, self._mu + 1)[:, newaxis])
        self.weights = self.weights / sum(self.weights)
        self.mueff = sum(self.weights) ** 2.0 / sum(self.weights ** 2)

        self.wcm_old = self.wcm
        self.xopt = self.wcm
        self.fopt = np.inf

        self.pc = zeros((dim, 1))
        self.ps = zeros((dim, 1))
        self.e_vector, self.e_value = eye(dim), ones((dim, 1))
        self.C = dot(self.e_vector, self.e_value * self.e_vector.T)
        self.invsqrt_C = dot(self.e_vector, self.e_value ** -1.0 * self.e_vector.T)

        # Strategy parameter: Adaptation
        # TODO: verify and update the strategy parameters
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        if self._mu == 1:
            self.c_1 = min([2, self._lambda / 3.0]) / ((self.dim + 1.3) ** 2 + self.mueff)
            self.damps = 0.3 + 2.0 * self.mueff / self._lambda + self.cs

        else:  # Original settings
            self.c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
            self.damps = (
                1.0 + 2 * np.max([0, sqrt((self.mueff - 1) / (self.dim + 1)) - 1]) + self.cs
            )
        self.c_mu = min(
            [
                1 - self.c_1,
                2 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff),
            ]
        )
        # TODO: verify this
        #    c_mu = 2 * (mueff-2.+1/mueff) / ((dim+2)**2+mueff)
        #    cc = 4.0 / (dim + 4.0)
        #    cs = (mueff+2.) / (dim+mueff+3.)

        # damps parameter tuning
        if "damps" in opts:
            self.damps = opts["damps"]
        else:
            # TODO: Parameter setting for mirrored orthogonal sampling
            # damps tuning for mirrored orthogonal sampling
            if self.sampling_method == 1:
                self.damps = 1.032 - 0.7821 * self.mueff / self._lambda + self.cs

            if self.sampling_method == 4 or self.sampling_method == 8:
                if 1 < 2:
                    # damps setting for small _lambda
                    self.damps = (
                        -0.6314 * (sqrt((self.mueff + 0.1572) / (self.dim + 1.647)) + 0.869)
                        + self.cs
                        + 1.49
                    )
                else:
                    # damps setting for large _lambda
                    self.damps = 1.17 + 4.625 * self.mueff / self._lambda - 2.704 * self.cs

            # Optimal damps setting for mirrored orthogonal sampling
            if self.sampling_method == 7:
                self.damps *= 0.3

        # Axuilirary variables
        self.chiN = dim ** 0.5 * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        self.aux = array([])

        # rescaling constant for derandomized step-size
        self.scale = self.chiN

        # Parameters for restart heuristics and warnings
        self.is_stop_on_warning = restart == "IPOP" or restart == "BIPOP"
        self.flg_warning = 0

        self.tolx = 1e-12 * self.sigma
        self.tolupx = 1e3 * max(self.sigma)
        self.tolfun = 1e-12
        self.nbin = int(10 + ceil(30.0 * self.dim / self._lambda))
        self.histfunval = zeros(self.nbin)

        # evolution history registration
        self.is_info_register = is_register
        if is_register:
            self.histsigma = zeros(self.eval_budget)
            self.hist_condition_number = zeros(self.eval_budget)
            self.hist_e_value = zeros((dim, self.eval_budget))
            self.hist_fbest = zeros(self.eval_budget)
            self.hist_xbest = zeros((self.eval_budget, self.dim))

            start = 200
            self.histindex = list(r_[0, linspace(start, self.eval_budget, 10)])
            self.histdist = zeros(size(self.histindex))
            self.ii = 0

        # if performaning pairwise selection
        self.is_pairwise_selection = self._mu != 1 and (
            self.sampling_method == 11
            or self.sampling_method == 4
            or self.sampling_method == 8
            or self.sampling_method == 1
            or self.sampling_method == 7
            or self.sampling_method == 44
        )

    def mutation(self):
        # ---------------------------- Mutation --------------------------------

        # Mirroring
        mode = self.sampling_method
        dim, _lambda, sigma, evalcount, scale, aux = (
            self.dim,
            self._lambda,
            self.sigma,
            self.evalcount,
            self.scale,
            self.aux,
        )

        if mode == 1 or mode == 11:
            if mod(evalcount + _lambda, 2) != 0:
                half = int(ceil(_lambda / 2.0))
                z = randn(dim, half)
                aux = -z[:, -1].reshape(-1, 1)
                z = append(z, -z[:, :-1], axis=1)
            else:
                half = int(floor(_lambda / 2.0))
                z = randn(dim, half)
                z = append(z, -z, axis=1)
                if len(aux) != 0:
                    z = append(aux, z, axis=1)
            self.half = half
        # Derandomized step size
        elif mode == 3:
            z = randn(dim, _lambda)
            z = scale * z / sqrt(sum(power(z, 2), 0))

        # Orthogonal mirrored sampling
        elif mode == 4 or mode == 7 or mode == 44:
            if mod(evalcount + _lambda, 2) != 0:
                half = int(ceil(_lambda / 2.0))
                z = zeros((dim, half))
                n = int(min([dim, half]))
                if dim < half:
                    z[:, dim:] = randn(dim, half - dim)
                q = qr(randn(dim, n))[0]
                l = chi.rvs(dim, size=n)
                z[:, 0:n] = l * q
                aux = -z[:, -1].reshape(-1, 1)
                z = append(z, -z[:, :-1], axis=1)
            else:
                half = int(floor(_lambda / 2.0))
                z = zeros((dim, half))
                n = int(min([dim, half]))
                if dim < half:
                    z[:, dim:] = randn(dim, half - dim)
                q = qr(randn(dim, n))[0]
                l = chi.rvs(dim, size=n)
                z[:, 0:n] = l * q
                z = append(z, -z, axis=1)
                if len(aux) != 0:
                    z = append(aux, z, axis=1)
            self.half = half
        # Orthogonal mirrored sampling...well
        elif mode == 8:
            if mod(evalcount + _lambda, 2) != 0:
                half = ceil(_lambda / 2.0)
                z = zeros((dim, half))
                n = min([dim, half])
                if dim < half:
                    z[:, dim:] = randn(dim, half - dim)
                tmp = randn(dim, n)
                l = sqrt(np.sum(power(tmp, 2), 0))
                q = qr(tmp)[0]
                z[:, 0:n] = l * q
                aux = -z[:, -1].reshape(-1, 1)
                z = append(z, -z[:, :-1], axis=1)
            else:
                half = floor(_lambda / 2.0)
                z = zeros((dim, half))
                n = min([dim, half])
                if dim < half:
                    z[:, dim:] = randn(dim, half - dim)
                tmp = randn(dim, n)
                l = sqrt(np.sum(power(tmp, 2), 0))
                q = qr(tmp)[0]
                z[:, 0:n] = l * q
                z = append(z, -z, axis=1)
                if len(aux) != 0:
                    z = append(aux, z, axis=1)
            self.half = half
        # Orthogonal sampling (random rotation)
        elif mode == 5:
            pass
            # z = dot(rand_orth_mat(dim), eye(dim))
            # n = dim
            # if dim > _lambda:
            #     p = arange(0, dim)
            #     shuffle(p)
            #     z = z[:, p[0:_lambda]]
            #     n = _lambda
            # l = chi.rvs(dim, size=n)
            # sign = rand(n)
            # sign[sign > .5] = 1
            # sign[sign <= .5] = -1
            # z = sign * l * z
            # if dim < _lambda:
            #     ss = randn(dim, _lambda-dim)
            #     z = append(z, ss, axis=1)

        # Orthogonal sampling (Gram-Schmidt)
        elif mode == 6:
            z = zeros((dim, _lambda))
            n = min([dim, _lambda])
            if dim < _lambda:
                z[:, dim:] = randn(dim, _lambda - dim)
            q = qr(randn(dim, n))[0]
            l = chi.rvs(dim, size=n)

        elif mode == 9:
            if mod(evalcount + _lambda, 2) != 0:
                half = int(ceil(_lambda / 2.0))
                z = randn(dim, half)
                q = randn(dim, half)
                p = array(
                    [
                        q[:, i] - inner(q[:, i], z[:, i]) * z[:, i] / norm(z[:, i]) ** 2
                        for i in range(half)
                    ]
                ).T
                p = (p / sqrt(sum(power(p, 2), 0))) * sqrt(sum(power(q, 2), 0))
                aux = p[:, -1].reshape(-1, 1)
                z = append(z, p[:, :-1], axis=1)
            else:
                half = int(floor(_lambda / 2.0))
                z = randn(dim, half)
                q = randn(dim, half)
                p = array(
                    [
                        q[:, i] - inner(q[:, i], z[:, i]) * z[:, i] / norm(z[:, i]) ** 2
                        for i in range(half)
                    ]
                ).T
                p = (p / sqrt(sum(power(p, 2), 0))) * sqrt(sum(power(q, 2), 0))
                z = append(z, p, axis=1)
                if len(aux) != 0:
                    z = append(aux, z, axis=1)
            self.half = half
        # Standard mutation
        else:
            z = randn(dim, _lambda)

        self.z = z
        self.offspring = add(self.wcm, sigma * self.e_vector.dot(self.e_value * self.z))

    def constraint_handling(self):
        self.offspring = boundary_handling(self.offspring, self.lb, self.ub)

    def evaluation(self):
        try:
            self.fitness = self.fitnessfunc(self.offspring)
        except Exception:  # if the fitness function evaluates a single point each time
            self.fitness = np.array([self.fitnessfunc(_) for _ in self.offspring.T])
        self.evalcount += self._lambda
        self.fitness_rank = argsort(self.fitness)
        self.fitness_true = self.fitness * (-1) ** (~self.is_minimize)

    def update(self):
        # Cumulation: Update evolution paths
        cc, cs, c_1, c_mu = self.cc, self.cs, self.c_1, self.c_mu
        wcm, wcm_old, mueff, invsqrt_C = self.wcm, self.wcm_old, self.mueff, self.invsqrt_C
        evalcount, _lambda = self.evalcount, self._lambda

        self.ps = (1 - cs) * self.ps + sqrt(cs * (2 - cs) * mueff) * dot(
            invsqrt_C, (wcm - wcm_old) / self.sigma
        )
        hsig = sum(self.ps ** 2.0) / (
            1.0 - (1.0 - cs) ** (2.0 * evalcount / _lambda)
        ) / self.dim < 2.0 + 4.0 / (self.dim + 1.0)
        self.pc = (1 - cc) * self.pc + hsig * sqrt(cc * (2.0 - cc) * mueff) * (
            wcm - wcm_old
        ) / self.sigma

        offset = (self.offspring[:, self.sel] - wcm_old) / self.sigma
        self.C = (
            (1.0 - c_1 - c_mu) * self.C
            + c_1 * (outer(self.pc, self.pc) + (1.0 - hsig) * cc * (2 - cc) * self.C)
            + c_mu * dot(offset, self.weights * offset.T)
        )
        # Adapt step size sigma
        self.sigma = self.sigma * exp((norm(self.ps) / self.chiN - 1) * self.cs / self.damps)

        if 11 < 3:  # TODO: Saw in the Hansen's codde, reason for this...
            self.sigma = self.sigma * exp(
                min([1, (norm(self.ps) / self.chiN - 1) * self.cs / self.damps])
            )

    def updateBD(self):
        # Eigen decomposition
        C = self.C  # lastest setting for
        C = triu(C) + triu(C, 1).T  # eigen decomposition
        if any(isinf(C)) > 1:  # interval
            self.flg_warning ^= 2 ** 0
        else:
            try:
                w, e_vector = eigh(C)
                e_value = sqrt(list(map(complex, w))).reshape(-1, 1)
                if any(~isreal(e_value)) or any(isinf(e_value)):
                    if self.is_stop_on_warning:
                        self.stop_dict["EigenvalueError"] = True
                    else:
                        self.flg_warning ^= 2 ** 1
                else:
                    self.e_value = real(e_value)
                    self.e_vector = e_vector
                    self.invsqrt_C = dot(e_vector, e_value ** -1 * e_vector.T)
            except LinAlgError:
                if self.is_stop_on_warning:
                    self.stop_dict["linalgerror"] = True
                else:
                    self.flg_warning ^= 2 ** 1

    def info_register(self):
        # Register historical info
        evalcount, _lambda, fitness, offspring = (
            self.evalcount,
            self._lambda,
            self.fitness,
            self.offspring,
        )
        if self.is_info_register:
            if self.ii < len(self.histindex):
                if abs(self.evalcount - self.histindex[self.ii]) <= self._lambda:
                    self.histdist[self.ii] = norm(self.wcm_old)
                    self.ii += 1
            self.histsigma[evalcount - _lambda : evalcount] = self.sigma
            self.hist_condition_number[evalcount - _lambda : evalcount] = (
                max(self.e_value) ** 2 / min(self.e_value) ** 2.0
            )
            self.hist_e_value[:, evalcount - _lambda : evalcount] = self.e_value
            self.hist_xbest[evalcount - _lambda : evalcount, :] = offspring[:, self.sel[0]]
            self.hist_fbest[evalcount - _lambda : evalcount] = fitness[self.sel[0]]

    def check_stop_criteria(self):
        # -------------------------- Restart criterion ------------------------------
        is_stop_on_warning = self.is_stop_on_warning
        sigma, evalcount, _lambda, fitness = self.sigma, self.evalcount, self._lambda, self.fitness

        self.stop_dict["ftarget"] = True if self.fopt <= self.f_target else False
        self.stop_dict["maxfevals"] = True if self.evalcount >= self.eval_budget else False

        if self.evalcount != 0:
            if np.any(fitness == inf) or np.any(fitness == np.nan):
                # TODO: nasty error to be debugged
                raise Exception("Somthing is wrong!")

            if (sigma < 1e-16) or (sigma > 1e6):
                self.flg_warning = True

            diagC = diag(self.C).reshape(-1, 1)
            self.histfunval[int(mod(evalcount / _lambda - 1, self.nbin))] = fitness[self.sel[0]]
            if (
                mod(evalcount, _lambda) == self.nbin
                and max(self.histfunval) - min(self.histfunval) < self.tolfun
            ):
                if is_stop_on_warning:
                    self.stop_dict["tolfun"] = True
                else:
                    self.flg_warning = True

            # Condition covariance
            if cond(self.C) > 1e14:
                if is_stop_on_warning:
                    self.stop_dict["conditioncov"] = True
                else:
                    self.flg_warning = True

            # TolX
            tmp = append(abs(self.pc), sqrt(diagC), axis=1)
            if all(self.sigma * (max(tmp, axis=1)) < self.tolx):
                if is_stop_on_warning:
                    self.stop_dict["TolX"] = True
                else:
                    self.flg_warning = True

            # TolUPX
            if any(sigma * sqrt(diagC)) > self.tolupx:
                if is_stop_on_warning:
                    self.stop_dict["TolUPX"] = True
                else:
                    self.flg_warning = True

            # No effective axis
            a = int(mod(evalcount / _lambda - 1, self.dim))
            if all(0.1 * sigma * self.e_value[a, 0] * self.e_vector[:, a] + self.wcm == self.wcm):
                if is_stop_on_warning:
                    self.stop_dict["noeffectaxis"] = True
                else:
                    sigma *= exp(0.2 + self.cs / self.damps)

            # No effective coordinate
            if any(0.2 * sigma * sqrt(diagC) + self.wcm == self.wcm):
                if is_stop_on_warning:
                    self.stop_dict["noeffectcoord"] = True
                else:
                    self.C += (self.c_1 + self.c_mu) * diag(
                        diagC * (self.wcm == self.wcm + 0.2 * sigma * sqrt(diagC))
                    )
                    sigma *= exp(0.05 + self.cs / self.damps)
            # Adjust step size in case of equal function values
            if (
                fitness[self.sel[0]]
                == fitness[self.sel[int(min([ceil(0.1 + _lambda / 4.0), self._mu - 1]))]]
            ):
                if is_stop_on_warning:
                    self.stop_dict["flatfitness"] = True
                else:
                    sigma *= exp(0.2 + self.cs / self.damps)

        # Handling warnings: Internally rectification of strategy paramters
        if self.flg_warning != 0:
            self.reset_state()
            self.flg_warning = False

    def reset_state(self):
        self.C = eye(self.dim)
        self.e_vector = eye(self.dim)
        self.e_value = ones((self.dim, 1))
        self.invsqrt_C = eye(self.dim)
        self.pc = zeros((self.dim, 1))
        self.ps = zeros((self.dim, 1))
        self.sigma = self.sigma0

    def reset_stop_dict(self):
        for key, _ in self.stop_dict.iteritems():
            self.stop_dict[key] = False

    def optimize(self):
        # TODO: use IPOP for now, to implement BIPOP method
        while self.restart_count < self.restart_budget:
            while True:
                self.info_register()
                self.mutation()
                self.constraint_handling()
                self.evaluation()

                # --------------------------- Comma selection ----------------------------
                # pairwise selection for mirroring
                if self.is_pairwise_selection:
                    #                    self.sel = pairwise_selection(self.fitness_rank, self.half, self._mu)
                    pass
                else:
                    self.sel = self.fitness_rank[0 : self._mu]

                # ------------------------- Weighted recombination ----------------------
                self.wcm_old = self.wcm
                self.wcm = dot(self.offspring[:, self.sel], self.weights)
                self.update()

                # update the eigenvectors and eigenvalues. for computational time concern
                if (
                    self.evalcount - self.eigeneval
                    > self._lambda / (self.c_1 + self.c_mu) / self.dim / 10
                ):
                    self.eigeneval = self.evalcount
                    self.updateBD()

                if self.fopt > self.fitness[self.sel[0]]:
                    self.fopt = self.fitness[self.sel[0]]
                    self.xopt = self.offspring[:, self.sel[0]].reshape(self.dim, -1)

                self.check_stop_criteria()

                if any(array(self.stop_dict.values()) == True):
                    break

            # if the termination criteria are met...
            if self.stop_dict["ftarget"] or self.stop_dict["maxfevals"]:
                break

            self.restart_count += 1
            self._lambda *= self.inc_popsize
            _mu_prime = (self._lambda - 1) / 2.0
            self._mu = int(ceil(_mu_prime))
            self.weights = log(_mu_prime + 1.0) - log(arange(1, self._mu + 1)[:, newaxis])
            self.weights = self.weights / sum(self.weights)
            self.mueff = sum(self.weights) ** 2.0 / sum(self.weights ** 2)

            self.wcm = rand(self.dim, 1) * (self.ub - self.lb) + self.lb
            self.reset_state()
            self.reset_stop_dict()

        return self.xopt, self.fopt, self.evalcount, self.stop_dict
