from __future__ import print_function
from itertools import combinations

import numpy as np
from scipy.special import binom
from myCounter import myCounter

class FreqOfFreq(object):
    def __init__(self, N, p_array):
        """E(N_i) and V(N_i) can be obtained using linearity of expected value
        and defining N_i = Sum_r I(n_r=i) where I(x) is the indecator function."""
        self.N = N
        if p_array.sum() != 1:
            raise ValueError("Args not sum properly: {} != 1".format(p_array.sum()))
        self.p_array = p_array[p_array>0]
        self.support = self.p_array.size


    def avgs(self):
        return [self.avg(_) for _ in np.arange(self.N + 1)]


    def vars(self):
        return [self.var(_) for _ in np.arange(self.N + 1)]


    def avg(self,i):
        if i>self.N:
            "This corner case may create 'divide by zero' errors."
            return 0.0
        return binom(self.N,i)*np.sum((self.p_array)**i*(1-self.p_array)**(self.N-i))


    def var(self,i):
        if i>self.N or self.support==1:
            "This corner case may create 'divide by zero' errors."
            return 0.0
        if self.support==2:
            "This corner case creates 0**0 problems in a."
            if self.N == 2*i:
                a = 2*multinomial(self.N,i,i,self.N-2*i)*np.prod(self.p_array)**i
            else:
                a = 0.0
        else:
            a = 0.0
            for pr, ps in combinations(self.p_array,2):
                a += (pr*ps)**i*(1-(pr+ps))**(self.N-2*i)
            a *= 2*multinomial(self.N,i,i,self.N-2*i)

        b = self.avg(i)
        c = b**2
        return a + b - c


    def mc_estimate(self,i):
        "This is written just to check the formulas used in avg and var."
        n_trials = 10000
        n_c = []
        space = np.arange(self.p_array.size)
        freqs = np.arange(max(self.N,i)+1)
        for _ in range(n_trials):
            pop = np.random.choice(space, size=self.N, p=self.p_array)
            fofs = myCounter(myCounter(pop, space).values(), freqs)
            fof = fofs[i]
            n_c.append(fof)
        return np.mean(n_c), np.var(n_c)


def multinomial(N,a,b,c):
    """Calculate multinomial coefficients using
    (a+b+c a b c) = (a+b b)*(a+b+c c)"""
    if N != a+b+c:
        raise ValueError("Args not sum properly: {} + {} + {} == {} != {} ".format(a,b,c,a+b+c,N))
    return binom(a+b, b)*binom(a+b+c,c)


if __name__=="__main__":
    def test(n_species, n_tries, p_array):
        with open("shit{}_{}.txt".format(n_species, n_tries),"w") as outFile:
            print("n_species:{} n_tries:{}".format(n_species, n_tries))
            print("N:{}".format(n_tries),file=outFile)
            print("p:{}".format(p_array))
            print("p:{}".format(p_array),file=outFile)
            print("",file=outFile)
            kk = FreqOfFreq(n_tries,p_array)
            for i in range(n_species+2):
                print("E({})={}  V({})={}".format(i, kk.avg(i), i, kk.var(i)))
                print("Mc estimation: {} +/- {}".format(*kk.mc_estimate(i)))
                print("")
            print("-----")

    for n_species,n_tries in [(1,1), (1,2), (3,2), (2,0), (2,1), (2,2), (2,3)]:
        p_array = np.ones(n_species)/n_species
        test(n_species, n_tries, p_array)
    for n_tries in range(3):
        test(3,n_tries,np.array([0.5,0.5,0.]))
