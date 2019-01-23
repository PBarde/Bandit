import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle

delta = 0.1
epsilon = 0.01
beta = 1.66


class Arms:
    def __init__(self, means, stds):
        self.n = len(means)
        idx = np.argsort(-means)
        self.means = np.asarray(means)[idx]
        self.stds = np.asarray(stds)[idx]

        H1 = 0
        for m in self.means[1:]:
            H1 += 1 / (self.means[0] - m) ** 2
        self.H1 = H1

    def sample(self, i):
        return self.stds[i] * np.random.randn(1) + self.means[i]


class UCB:
    def __init__(self, n, arms):
        self.n = n
        self.estimated_means = np.zeros(n)
        self.n_pulls = np.zeros(n)
        self.pull_tot = 0
        self.alpha = ((2 + beta) / beta) ** 2 * (
                1 + np.log(2 * np.log(((2 + beta) / beta) ** 2 * n / delta)) / np.log(n / delta))
        self.previous_pulls = [deque(maxlen=n) for _ in range(n)]
        self.Pi = []
        self.arms = arms
        self.recorded_proportions = []

    def U(self, t, d, T):
        # return (1 + epsilon ** 0.5) * (
        #             (2*(1 + epsilon) * np.log((np.log((1 + epsilon) * T)) / d)) / t) ** 0.5
        # return (1 + epsilon ** 0.5) * (((1 + epsilon) * np.log(T * (np.log((1 + epsilon) * T) + 2) / d)) / (2 * t)) ** 0.5
        # return (1 + epsilon ** 0.5) * (
        #             ((1 + epsilon) * T * np.log( (np.log((1 + epsilon) * T) + 2) / d)) / (2 * t**2)) ** 0.5
        # return (1 + epsilon ** 0.5) * (((1 + epsilon) * np.log((np.log((1 + epsilon) * t) + 2) / d)) / (2 * t)) ** 0.5
        # return (1 + epsilon ** 0.5) * (((1 + epsilon) * t * np.log((np.log((1 + epsilon) * t) + 2) / d)) / (2 * t)) ** 0.5
        return (1 + epsilon ** 0.5) * (
                    ((1 + epsilon) * np.log(t * (np.log((1 + epsilon) * t) + 2) / d)) / (2 * t)) ** 0.5
    def C(self, ti):
        return (1 + beta) * self.U(ti, delta / self.n, self.pull_tot)
        # return (1 + beta) * self.U(ti, delta, self.pull_tot)

    def pull(self, i):
        ri = self.arms.sample(i)
        self.n_pulls[i] += 1
        # self.pull_tot += 1
        self.update_mean(i, ri)

    def update_pulls(self, i):
        for j, p in enumerate(self.previous_pulls):
            if j == i:
                p.append(1)
            else:
                p.append(0)

    def update_mean(self, i, ri):
        # if self.n_pulls[i] == 0:
        #     self.estimated_means[i]=
        self.estimated_means[i] += 1 / self.n_pulls[i] * (ri - self.estimated_means[i])

    def record(self):
        summed_previous_pulls = []

        for pulls in self.previous_pulls:
            summed_previous_pulls.append(sum(pulls))

        self.recorded_proportions.append(np.asarray(summed_previous_pulls) / sum(summed_previous_pulls))

    def get_idx_to_pull(self):
        values = []
        for mi, ti in zip(self.estimated_means, self.n_pulls):
            values.append(mi + self.C(ti))
        idx_to_pull = np.argmax(values)
        return idx_to_pull

    def stopping_condition(self):
        for i, ti in enumerate(self.n_pulls):
            summed_tj = 0
            for j, tj in enumerate(self.n_pulls):
                if not i == j:
                    summed_tj += tj

            if ti > self.alpha * summed_tj:
                return True

        return False

    def step(self):
        idx_to_pull = self.get_idx_to_pull()
        self.pull(idx_to_pull)
        self.update_pulls(idx_to_pull)
        self.pull_tot += 1
        self.record()
        return self.stopping_condition()

    def initialize(self):
        for i in range(self.n):
            self.pull(i)
            self.pull_tot += 1
            self.update_pulls(i)

    def search(self):
        stopping_condition = False
        self.initialize()
        while not stopping_condition:
            stopping_condition = self.step()
        return self.recorded_proportions


class LUCB(UCB):
    def __init__(self, means, stds):
        super().__init__(means, stds)
        self.alpha = None

    def U(self, t, d, T):
        # return (1 + epsilon ** 0.5) * (
        #             (2*(1 + epsilon) * np.log((np.log((1 + epsilon) * T)) / d)) / t) ** 0.5
        # return (1 + epsilon ** 0.5) * (((1 + epsilon) * np.log(T * (np.log((1 + epsilon) * T) + 2) / d)) / (2 * t)) ** 0.5
        # return (1 + epsilon ** 0.5) * (
        #             ((1 + epsilon) * T * np.log((np.log((1 + epsilon) * T) + 2) / d)) / (2 * t**2)) ** 0.5
        return (1 + epsilon ** 0.5) * (((1 + epsilon) * np.log(t*(np.log((1 + epsilon) * t) + 2) / d)) / (2 * t)) ** 0.5
        # return (1 + epsilon ** 0.5) * (((1 + epsilon) * t * np.log((np.log((1 + epsilon) * t) + 2) / d)) / (2 * t)) ** 0.5
        # return (1 + epsilon ** 0.5) * (((1 + epsilon) * np.log((np.log((1 + epsilon) * t) + 2) / d)) / (2 * t)) ** 0.5
    def C(self, ti):
        return self.U(ti, delta / self.n, self.pull_tot)
        # return self.U(ti, delta, self.pull_tot)

    def get_idx_to_pull(self):
        ht_idx = np.argmax(self.estimated_means)

        idx = 0
        lt_max = -np.inf
        lt_idx = None
        for mi, ti in zip(self.estimated_means, self.n_pulls):
            if not idx == ht_idx:
                lt_val = mi + self.C(ti)
                if lt_val > lt_max:
                    lt_max = lt_val
                    lt_idx = idx
            idx += 1

        return [ht_idx, lt_idx]

    def update_pulls(self, set_of_idx):
        if isinstance(set_of_idx, list):
            set_of_idx = set(set_of_idx)
        else:
            set_of_idx = set([set_of_idx])
        for j, p in enumerate(self.previous_pulls):
            if j in set_of_idx:
                p.append(1)
            else:
                p.append(0)

    def step(self):
        idxes_to_pull = self.get_idx_to_pull()
        for idx in idxes_to_pull:
            self.pull(idx)
        self.update_pulls(idxes_to_pull)
        self.pull_tot += 1
        self.record()
        return self.stopping_condition(idxes_to_pull)

    def stopping_condition(self, idxes_to_pull):
        i_ht, i_lt = idxes_to_pull
        mu_ht, t_ht = self.estimated_means[i_ht], self.n_pulls[i_ht]
        mu_lt, t_lt = self.estimated_means[i_lt], self.n_pulls[i_lt]

        if (mu_ht - self.C(t_ht)) > (mu_lt + self.C(t_lt)):
            return True
        else:
            return False


class AE(LUCB):
    def __init__(self, means, stds):
        super().__init__(means, stds)
        self.to_pull = [i for i in range(self.n)]

    def U(self, t, d, T):
        # return (1 + epsilon ** 0.5) * (
        #             ((1 + epsilon) * np.log((np.log((1 + epsilon) * t) + 2) / d)) / (2 * t)) ** 0.5
        return (1 + epsilon ** 0.5) * (
                ((1 + epsilon) * np.log(t*(np.log((1 + epsilon) * t) + 2) / d)) / (2 * t)) ** 0.5
    def C(self, ti):
        return 2 * self.U(ti, delta / self.n, self.pull_tot)

    def get_idx_to_pull(self):
        idx = 0
        max_v = -np.inf
        max_idx = None
        for mi, ti in zip(self.estimated_means, self.n_pulls):
            v = mi + self.C(ti)
            if v > max_v:
                max_v = v
                max_idx = idx
            idx += 1

        to_pull = []
        v_low = self.estimated_means[max_idx] - self.C(self.n_pulls[max_idx])

        idx = 0
        for mi, ti in zip(self.estimated_means, self.n_pulls):
            v = mi + self.C(ti)
            if v_low < v:
                to_pull.append(idx)
            idx += 1

        return to_pull

    def stopping_condition(self, idxes_to_pull):
        self.to_pull = idxes_to_pull
        return len(idxes_to_pull) == 1

if __name__ == '__main__':
    n = 6
    n_trials = 2000
    means = np.arange(1, -1 / 5, -1 / 5)
    stds = [1 / 2] * n
    arms = Arms(means, stds)
    trials_prop = []
    best_arm = []
    alg = 'AE'
    for r in range(n_trials):
        if alg == 'LUCB':
            algo = LUCB(n, arms)
        elif alg == 'UCB':
            algo = UCB(n, arms)
        elif alg == 'AE':
            algo = AE(n, arms)
        else:
            raise ValueError

        rec = algo.search()
        trials_prop.append(rec)

        if alg in {'UCB','LUCB'}:
            best_arm.append(np.argmax(algo.estimated_means))
        elif alg == 'AE':
            best_arm.append(algo.to_pull[0])
        print(f'run : {r}')

    # fp = open(f'./results/{alg}_{n_trials}_raw.pckl', 'wb')
    # pickle.dump(trials_prop, fp)
    # fp.close()

    avg_props = []
    for i in range(n):
        avg = []
        max_len = max([len(trial) for trial in trials_prop])
        for t in range(max_len):
            avg_t = 0
            nv = 0
            for tt, trial in enumerate(trials_prop):
                # if len(trial) < t + 1: continue
                if len(trial) < t + 1:
                    if i == best_arm[tt]:
                        val = 1.
                    else:
                        val = 0.
                else:
                    val = trial[t][i]
                nv += 1
                try:
                    avg_t += 1 / nv * (val - avg_t)
                except:
                    raise
            avg.append(avg_t)
        avg_props.append(avg)

    fp = open(f'./results/{alg}_{n_trials}_means.pckl', 'wb')
    pickle.dump(avg_props, fp)
    fp.close()

    cmap = plt.get_cmap('tab20')

    t_steps = np.arange(len(avg_props[0])) / arms.H1
    for i, avgs in enumerate(avg_props):
        plt.plot(t_steps, avgs, color=cmap(i), label=f'mu_{i}')
    plt.xlabel('Number of pulls (units of H1)')
    plt.ylabel('P(I_t = i)')
    # plt.xlim(0, 75)
    plt.legend()
    plt.show()

    # print(arms.H1)
    # print(trials_prop)
    # print(algo.estimated_means)
    # print(arms.means)
    # print(algo.n_pulls)
    # print(algo.alpha)
    # print(sum(trials_prop[0][-1]))

    # todo: pull arm, update and stopping condition and reccording
