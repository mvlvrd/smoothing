from collections import Counter
import numpy as np
from scipy.stats import chi2, linregress
from FreqOfFreq import FreqOfFreq

"""Assume in_dict is a dictionary with all possible events as keys,
including those with zero counts."""

def laplace(in_dict, alpha=None):
    """When alpha=1 this is the Laplace smoothing.
    When alpha=0.5 it is called Expected Likelihood Estimate (ELE).
    When alpha=0 it is the Maximum Likelihood Estimator."""
    if alpha is None:
        alpha = 1.
    vals = np.array(in_dict.values())
    corr_counts = np.sum(vals) + alpha*np.size(vals)
    return {key: (val + alpha)/corr_counts for key, val in in_dict.items()}

def steinhaus(in_dict):
    """Steinhaus estimation as discussed in
    H.Steinhaus Ann.Math.Statist. Vol.28 3 (1957) 633-648"""
    steinhaus_coeff = np.sqrt(np.sum(np.array(in_dict.values())))/len(in_dict)
    return laplace(in_dict, alpha=steinhaus_coeff)

def simple_good_turing(in_dict):
    fof = Counter(in_dict.values())

    N = sum(in_dict.values())
    n_unseen = sum([1 for v in in_dict.values() if v==0])

    gt_dict = sgt(fof)

    "Renormalize gt_dict. p_0 is the total probability of the unobserved species"
    norm_gt_dict = dict()
    if n_unseen > 0:
        p_0 = gt_dict[0]/N
    else:
        p_0 = 0.0
    norm_factor = (1. - p_0)/sum([gt_dict[v] for v in in_dict.itervalues() if v > 0])
    for k, v in in_dict.iteritems():
        if v==0:
            norm_gt_dict[k] = p_0/n_unseen
        else:
            norm_gt_dict[k] = gt_dict[v]*norm_factor

    return norm_gt_dict


def sgt(fof):
    """Returns an array g_t with indexes the frequencies 0<=r<Max_freq+1 and
    values the smoothed values r* for each index.
    Notice that g_t[0] is not 0* but E_{N_0},
    the expected value of all unobserved species."""
    freqs, nr = zip(*sorted(fof.iteritems(), key=lambda x: x[0]))

    "Use averaging transformation for freq of freqs."
    d = np.concatenate(([1], np.diff(freqs)))
    d = np.concatenate(((d[1:] + d[:-1])/2., [d[-1]]))
    znr = nr/d
    slope, intercept, r_value, p_value, std_err = linregress(x=np.log10(freqs), y=np.log10(znr))

    fmax_plus1 = max(freqs) + 1

    "Check slope < -1:"
    if slope > -1:
        raise ValueError("Bad slope in Simple Good-Turing smoothing:{}.".format(slope))
    else:
        smoothed_fof = (10**intercept)*np.arange(fmax_plus1 + 1)**slope

    gt_dict = np.zeros(fmax_plus1)
    behind_break_point = True
    for val in range(1,fmax_plus1):
        lgt_estimate = (val+1.)*smoothed_fof[val+1]/smoothed_fof[val]
        if behind_break_point and val in fof and (val+1) in fof:
            t_estimate = (val+1.)*fof[val+1]/fof[val]
            var_val = (val+1.)/fof[val]*np.sqrt(fof[val+1]*(1. + fof[val+1]/float(fof[val])))
            if abs(t_estimate - lgt_estimate) > 1.65*var_val:
                gt_dict[val] = t_estimate
            else:
                behind_break_point = False
                gt_dict[val] = lgt_estimate
        else:
            if behind_break_point: print("eh",val)
            behind_break_point = False
            gt_dict[val] = lgt_estimate

    gt_dict[0] = smoothed_fof[1]
    return gt_dict


def good_turing(in_dict):
    counts = sum(in_dict.vals())
    p_array = np.array(in_dict.vals(), dtype=np.float64)/counts
    fof = FreqOfFreq(counts, p_array)
    fof_vect, fof_vars = fof.avgs(), fof.vars()

    """Smooth the frequency of frequencies array before applying GT(r) = (r+1)*N_{r+1}/N_r.
    smoothed_fof is assumed to be normalized."""
    smoothed_fof = _smooth(fof_vect, fof_vars)
    gt_dict = {key: (val+1)*smoothed_fof[val+1]/(smoothed_fof[val]*counts) for key, val in in_dict.iteritems()}
    return gt_dict


def _smooth(fof_vect, fof_vars): #Incomplete
    "inputs must be 0-N arrays. It returns a 0-N+1 array."
    N = fof_vect.size
    slope, intercept = _weighted_linregress(y=fof_vect, sigma2=fof_vars)
    lin_smooth = slope*np.arange(N+2) + intercept

    """This is not implemented still because I do not understand (Eq. 19) properly:
    Why \Sum_{t=1}^r and not \Sum_{t=0}^r ?"""
    #Check that the \Chi_2 is not too large (Eq.19)
    df = N + 1
    threshold = 0.90
    chi2_obs = np.sum((lin_smooth[:-1] - fof_vect)**2/fof_vars)
    cdf = chi2.cdf(chi2_obs, df=df)[0]
    if cdf >= threshold:
        lin_smooth = fof_vect

    return lin_smooth


def _weighted_linregress(x=None, y=None, sigma2=None):
    if x is None:
        x = np.arange(y.size)
    if y is None:
        raise ValueError('Array y must be initialized.')
    if sigma2 is None:
        w2 = np.ones(x.size, dtype=np.int8)
    else:
        w2 = 1./sigma2
    sx, sy = np.sum(w2*x), np.sum(w2*y)
    sxx, sxy, s1 = np.sum(w2*x**2), np.sum(w2*x*y), np.sum(w2)

    slope = (sxy - sx*sy/s1)/(sxx - sx*sx/s1)
    intercept = (sy - slope*sx)/s1
    return slope, intercept
