
import scipy
from scipy.stats import weibull_min
import scipy.optimize
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


c_init = [0.1,0.5,1,5,10,20,50,100]
plot_res = None

# start from function get_lipschitz_estimate()
def plot_weibull(sample, c, loc, scale, ks, pVal, p, q, figname):
# def plot_weibull(sample, loc, scale, ks, pVal, p, q, figname):
    # compare the sample histogram and fitting result
    fig, ax = plt.subplots(1, 1)

    x = np.linspace(-1.001 * max(sample), -0.999 * min(sample), 1000);
    # x = np.linspace(0.99 * min(sample), 1.01 * max(sample), 100)
    # plot the Theoretical pdf red line image
    ax.plot(x, weibull_min.pdf(x, c, loc, scale), 'r-', label='fitted pdf ' + p + '-bnd')


    ax.hist(-sample, density=True, bins=100, histtype='stepfilled')
    # ax.hist(sample, density=True, bins=20, histtype='stepfilled')
    ax.legend(loc='best', frameon=False)
    # plt.xlabel('Lips_' + q)
    plt.xlabel('-Lips_' + q)
    plt.ylabel('pdf')
    plt.title('c = {:.2f}, loc = {:.2f}, scale = {:.2f}, ks = {:.2f}, pVal = {:.2f}'.format(c, loc, scale, ks, pVal))
    # plt.title('loc = {:.2f}, scale = {:.2f}, ks = {:.2f}, pVal = {:.2f}'.format(loc, scale, ks, pVal))
    plt.savefig(figname)
    plt.close()

def fit_and_test(rescaled_sample, sample, loc_shift, shape_rescale, optimizer, c_i):
    try:
        [c, loc, scale] = weibull_min.fit(-rescaled_sample, c_i, optimizer=optimizer)
    except:
        print(rescaled_sample)
        [c, loc, scale] = weibull_min.fit(-rescaled_sample, c_i, optimizer=optimizer)
    # [loc, scale] = gumbel_l.fit(rescaled_sample, optimizer=optimizer)
    # [c, loc, scale] = genextreme.fit(rescaled_sample, c_i, optimizer=optimizer)
    #
    loc = - loc_shift + loc * shape_rescale
    scale *= shape_rescale
    ks, pVal = scipy.stats.kstest(-sample.ravel(), 'weibull_min', args = (c, loc, scale))
    # ks, pVal = scipy.stats.kstest(sample, 'genextreme', args = (c, loc, scale))
    # ks, pVal = scipy.stats.kstest(sample, 'gumbel_l', args=(loc, scale))

    return c, loc, scale, ks, pVal
    # return loc, scale, ks, pVal

def get_best_weibull_fit(sample, use_reg=False, shape_reg=0.01):
    # initialize dictionary to save the fitting results

    fitted_paras = {"c": [], "loc": [], "scale": [], "ks": [], "pVal": []}
    # reshape the data into a better range
    # this helps the MLE solver find the solution easier
    loc_shift = np.amax(sample)
    dist_range = np.amax(sample) - np.amin(sample)
    print(f'Max of delta_g is {np.amax(sample)} and the Min of delta_g is {np.amin(sample)}')
    if dist_range > 2.5:
        shape_rescale = dist_range
    else:
        shape_rescale = 1.0
    print("shape rescale = {}".format(shape_rescale))
    rescaled_sample = np.copy(sample)
    rescaled_sample = rescaled_sample.astype(np.float64)
    rescaled_sample -= loc_shift
    rescaled_sample /= shape_rescale

    print("loc_shift = {}".format(loc_shift))
    ##print("rescaled_sample = {}".format(rescaled_sample))

    # fit weibull distn: sample follows reverse weibull dist, so -sample follows weibull distribution

    results = []
    for c_i in c_init:
        result = fit_and_test(rescaled_sample, sample, loc_shift, shape_rescale, scipy.optimize.fmin, c_i)
        results.append(list(result))

    # with Pool(processes=4) as pool:
    #     results = pool.map(
    #         partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale, scipy.optimize.fmin), c_init)

    for res, c_i in zip(results, c_init):
        c = res[0]
        loc = res[1]
        scale = res[2]
        ks = res[3]
        pVal = res[4]

        # loc = res[0]
        # scale = res[1]
        # ks = res[2]
        # pVal = res[3]
        print(
            "[DEBUG][L2] c_init = {:5.5g}, fitted c = {:6.2f}, loc = {:7.2f}, scale = {:7.2f}, ks = {:4.2f}, pVal = {:4.2f}, max = {:7.2f}".format(
                c_i, c, loc, scale, ks, pVal, loc_shift))
        # print(
        #     "[DEBUG][L2] c_init = {:5.5g}, loc = {:7.2f}, scale = {:7.2f}, ks = {:4.2f}, pVal = {:4.2f}, max = {:7.2f}".format(
        #         c_i, loc, scale, ks, pVal, loc_shift))

        ## plot every fitted result
        # plot_weibull(sample,c,loc,scale,ks,pVal,p)

        fitted_paras['c'].append(c)
        fitted_paras['loc'].append(loc)
        fitted_paras['scale'].append(scale)
        fitted_paras['ks'].append(ks)
        fitted_paras['pVal'].append(pVal)

    # get the paras of best pVal among c_init
    max_pVal = np.nanmax(fitted_paras['pVal'])
    # if np.isnan(max_pVal) or max_pVal < 0.001:
    #     print("ill-conditioned samples. Using maximum sample value.")
    #     # handle the ill conditioned case
    #     return -1, -1, -max(sample), -1, -1, -1

    max_pVal_idx = fitted_paras['pVal'].index(max_pVal)

    c_init_best = c_init[max_pVal_idx]
    c_best = fitted_paras['c'][max_pVal_idx]
    loc_best = fitted_paras['loc'][max_pVal_idx]
    scale_best = fitted_paras['scale'][max_pVal_idx]
    ks_best = fitted_paras['ks'][max_pVal_idx]
    pVal_best = fitted_paras['pVal'][max_pVal_idx]

    # pool.close()
    # pool.join()
    return c_init_best, c_best, loc_best, scale_best, ks_best, pVal_best
    # return c_init_best, loc_best, scale_best, ks_best, pVal_best

def get_lipschitz_estimate(G_max, figname, norm = "L2", use_reg = False, shape_reg = 0.01):
    global plot_res
    G_max = np.array(G_max)
    # mean = np.mean(G_max)
    # std_dev = np.std(G_max)
    # z_scores = [(x, (x - mean) / std_dev) for x in G_max]
    # filtered_data = [x[0] for x in z_scores if abs(x[1]) >= 3]
    # filtered_data_array = np.array(filtered_data)
    # to_keep = np.isin(G_max, filtered_data_array)
    # data_without_filtered = G_max[~to_keep]
    # G_max = copy.deepcopy(data_without_filtered)

    c_init, c, loc, scale, ks, pVal = get_best_weibull_fit(G_max, use_reg, shape_reg)
    # c_init, loc, scale, ks, pVal = get_best_weibull_fit(G_max, use_reg, shape_reg)

    # the norm here is Lipschitz constant norm, not the bound's norm
    if norm == "L1":
        p = "i"; q = "1"
    elif norm == "L2":
        p = "2"; q = "2"
    elif norm == "Li":
        p = "1"; q = "i"
    else:
        print("Lipschitz norm is not in 1, 2, i!")

    if plot_res is not None:
        plot_res.get()

    # plot_weibull(G_max,c,loc,scale,ks,pVal,p,q,figname)
    # draw the K-S results
    if figname:
        figname = figname + '_ '+ "L "+ p + ".png"
        # with Pool(processes=4) as pool:
        #     # plot_res = plot_weibull(G_max ,c ,loc ,scale ,ks ,pVal ,p ,q ,figname)
        #     plot_res = pool.apply_async(plot_weibull, (G_max, c, loc, scale, ks, pVal, p, q, figname))
        plot_res = plot_weibull(G_max ,c, loc ,scale ,ks ,pVal ,p ,q ,figname)
        # plot_res = pool.apply_async(plot_weibull, (G_max ,c ,loc ,scale ,ks ,pVal ,p ,q ,figname))
        # pool.close()
        # pool.join()
    return {'Lips_est' :-loc, 'shape' :c, 'loc': loc, 'scale': scale, 'ks': ks, 'pVal': pVal}
    # return {'Lips_est' :-loc, 'loc': loc, 'scale': scale, 'ks': ks, 'pVal': pVal}
    # return np.max(G_max)

#######################################################################################################################

def plot_one_distribution(sample, q, figname):
    # compare the sample histogram and fitting result
    fig, ax = plt.subplots(1, 1)
    # x = np.linspace(0.99 * min(sample), 1.01 * max(sample), 100)
    # plot the Theoretical pdf red line image
    ax.set_xlim([0.99 * min(sample), 1.01 * max(sample)])
    # ax.plot(x, gumbel_l.pdf(x, loc, scale), 'r-', label='fitted pdf ' + p + '-bnd')

    ax.hist(sample, density=True, bins=100, histtype='stepfilled', label='Lips_' + q)
    ax.legend(loc='best', frameon=False)
    plt.xlabel('Lips_' + q)

    plt.ylabel('pdf')

    plt.title('One-point Distribution')
    plt.savefig(figname)
    plt.close()


def get_lipschitz_one_distribution(G_max, norm = "L2", figname = "one_point_distribution", use_reg = False, shape_reg = 0.01):
    global plot_res
    G_max = np.array(G_max)

    # the norm here is Lipschitz constant norm, not the bound's norm
    if norm == "L1":
        p = "i"; q = "1"
    elif norm == "L2":
        p = "2"; q = "2"
    elif norm == "Li":
        p = "1"; q = "i"
    else:
        print("Lipschitz norm is not in 1, 2, i!")

    # if plot_res is not None:
    #     plot_res.get()

    # plot_weibull(G_max,c,loc,scale,ks,pVal,p,q,figname)
    # draw the K-S results
    if figname:
        figname = figname + '_ '+ "L "+ p + ".png"
        plot_res = plot_one_distribution(G_max,q ,figname)
        # plot_res = pool.apply_async(plot_weibull, (G_max ,c ,loc ,scale ,ks ,pVal ,p ,q ,figname))
        # pool.close()
        # pool.join()

    return
    # return np.max(G_max)