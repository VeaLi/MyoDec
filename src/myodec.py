import warnings

warnings.filterwarnings('ignore')
import os
import copy

import numpy as np
# np.set_printoptions(legacy='1.25')
np.random.seed(7)

import pandas as pd

from matplotlib import colors as matplotlib_colors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq
from scipy import signal as scipy_signal
from scipy import linalg


from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA


import pywt

# from numba import jit
# from numba import int32, float32, float64


def silhouette_score(S, peak_indices):

    # Create clusters
    peak_cluster = S[peak_indices]
    noise_cluster = np.delete(S, peak_indices)

    # Create centroids
    peak_centroid = peak_cluster.mean()
    noise_centroid = noise_cluster.mean()

    # Calculate within-cluster sums of point-to-centroid distances
    intra_sums = (abs(peak_cluster - peak_centroid).sum() +
                  abs(noise_cluster - noise_centroid).sum())

    # Calculate between-cluster sums of point-to-centroid distances
    inter_sums = (abs(peak_cluster - noise_centroid).sum() +
                  abs(noise_cluster - peak_centroid).sum())

    diff = inter_sums - intra_sums

    sil = diff / max(intra_sums, inter_sums)

    return sil


def get_mu_ts_sil_pnr(signal,
                      cluster_method,
                      PEAK_DISTANCE,
                      pca_object=None,
                      agg_object=None):

    if cluster_method == 'agglomerative':
        # keep negative flat
        signal_positive = signal
    elif cluster_method == 'k-means':
        # square
        signal_positive = signal**2

    # find peaks, neighbor method
    peaks_ts, _ = scipy_signal.find_peaks(signal_positive,
                                          height=0,
                                          distance=PEAK_DISTANCE)

    peaks_positive = signal_positive[peaks_ts]
    if len(peaks_positive) == 0:
        return [], 0, 0, 0, 0

    if cluster_method == 'agglomerative':

        signal_positive_extended = extend2(signal_positive.reshape(1, -1),
                                           int(PEAK_DISTANCE))
        peaks_positive_extended = signal_positive_extended[:, peaks_ts].T

        try:
            peaks_positive_extended = pca_object.fit_transform(
                peaks_positive_extended)
        except:
            return [], 0, 0, 0, 0

        clustering = agg_object.fit(peaks_positive_extended)

        peaks_labels = clustering.labels_

        centroids = [
            peaks_positive[~peaks_labels.astype('bool')].mean(),
            peaks_positive[peaks_labels.astype('bool')].mean()
        ]
        mu_ts = peaks_ts[peaks_labels == np.argmax(centroids).astype('int')]
        mu_ts = np.delete(mu_ts, np.argwhere(signal[mu_ts] < 0).reshape(-1))

    elif cluster_method == 'k-means':

        kmeans = KMeans(n_clusters=2,
                        init="k-means++",
                        max_iter=500,
                        n_init=10,
                        random_state=7).fit(peaks_positive.reshape(-1, 1))

        peaks_labels = kmeans.predict(peaks_positive.reshape(-1, 1))

        centroids = [
            peaks_positive[~peaks_labels.astype('bool')].mean(),
            peaks_positive[peaks_labels.astype('bool')].mean()
        ]
        mu_ts = peaks_ts[peaks_labels == np.argmax(centroids).astype('int')]
        mu_ts = np.delete(mu_ts, np.argwhere(signal[mu_ts] < 0).reshape(-1))

    ###

    if len(mu_ts) == 0:
        return [], 0, 0, 0, 0

    # "false" mu peaks are the rest peaks or rest of signal
    # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters
    SIL0 = silhouette_score(signal_positive, mu_ts)
    SIL1 = silhouette_score(
        peaks_positive,
        np.arange(len(peaks_positive))[peaks_labels.astype('bool')])

    signal_positive = signal_positive**2
    not_mu_ts = np.delete(np.arange(len(signal_positive)), mu_ts)
    pnr_numerator = signal_positive[mu_ts].mean()
    pnr_denominator = signal_positive[not_mu_ts].mean()
    PNR0 = 10 * np.log10(pnr_numerator / pnr_denominator)

    not_mu_ts = np.array([i for i in peaks_ts if i not in mu_ts])
    pnr_numerator = signal_positive[mu_ts].mean()
    pnr_denominator = signal_positive[not_mu_ts].mean()
    PNR1 = 10 * np.log10(pnr_numerator / pnr_denominator)

    return mu_ts, SIL0, SIL1, PNR0, PNR1


def get_mu_avg_shape_win(signal, mu_ts, win=50, recenter=False):

    def pad0(muap, win):
        while len(muap) < win:
            muap = np.append(muap, 0.0)

        return muap

    if win % 2 != 0:
        win += 1

    winh = win // 2

    MUAPS = []
    INDS = []
    for ts in mu_ts:
        ts_old = ts
        l, r = max(0, ts - winh), min(ts + winh, len(signal) - 1)
        if recenter:
            index = np.arange(l, r)
            ts = index[np.argmax(signal[l:r])]
            l, r = max(0, ts - winh), min(ts + winh, len(signal) - 1)
        muap = pad0(signal[l:r], win)
        INDS.append(np.arange(l, r))

        assert len(muap) == win
        MUAPS.append(muap)

    avg_muap = np.array(MUAPS).mean(axis=0)

    fit = np.zeros_like(signal)
    for ind in INDS:
        fit[ind] = avg_muap[:len(ind)]

    return avg_muap, np.array(MUAPS), fit


def whiten(XT, whiten_solver='svd'):
    n_samples = XT.shape[1]
    # Centering the features of X
    X_mean = XT.mean(axis=-1)
    XT -= X_mean[:, np.newaxis]

    # Whitening and preprocessing by PCA
    if whiten_solver == "eigh":
        # Faster when num_samples >> n_features
        d, u = linalg.eigh(XT.dot(XT.T))
        sort_indices = np.argsort(d)[::-1]
        eps = np.finfo(d.dtype).eps * 10
        degenerate_idx = d < eps
        d[degenerate_idx] = eps  # For numerical issues
        np.sqrt(d, out=d)
        d, u = d[sort_indices], u[:, sort_indices]
    elif whiten_solver == "svd":
        u, d = linalg.svd(XT, full_matrices=False, check_finite=False)[:2]

    # Give consistent eigenvectors for both svd solvers
    u *= np.sign(u[0])

    K = (u / d).T  #[:n_components]  # see (6.33) p.140
    del u, d
    X1 = np.dot(K, XT)
    X1 *= np.sqrt(n_samples)
    return X1


def extend(x, R):
    # The extension of the HD sEMG signals is a strategy to increase the number of observations and then,
    # theoretically, to increase the number of estimated sources.

    ch, tm = x.shape
    if R > 0:
        R += 1
        x_ = np.zeros((ch * R, tm))
        for c in range(ch):
            for r in range(R):
                x_[(r + c * R), r:] = x[c, :(tm - r)]
        return x_
    else:
        return x


def extend2(x, R):
    # The extension of the HD sEMG signals is a strategy to increase the number of observations and then,
    # theoretically, to increase the number of estimated sources.

    ch, tm = x.shape
    if R > 0:
        if R % 2 == 1:
            R += 1

        HR = R // 2

        x_ = np.zeros(((ch * R) - 1, tm))
        for c in range(ch):
            ri = 0

            for r in reversed(range(1, HR)):
                x_[(ri + c * R), :(tm - r)] = x[c, r:]
                ri += 1

            for r in range(HR):
                x_[(ri + c * R), r:] = x[c, :(tm - r)]
                ri += 1

        return x_
    else:
        return x


def drop_duplicates(sources_info, amount=0.3, delta=1):
    drop = []

    for i, src1 in enumerate(sources_info):
        for j, src2 in enumerate(sources_info):
            if i == j:
                continue

            if j < i:
                continue

            if i in drop or j in drop:
                continue

            s1 = src1['mu_ts']
            s2 = src2['mu_ts']

            M = min(len(s1) * amount, len(s2) * amount)

            # l = len(['' for e1 in s1 for e2 in s2 if abs(e1 - e2) <= delta])

            l = 0
            for e1 in s1:
                for e2 in s2:
                    if abs(e1 - e2) <= delta:
                        l += 1
                        continue

                    if e2 > e1:
                        continue

                if l >= M:
                    break

            if l >= M:
                if src1['SIL1'] >= src2['SIL1']:
                    drop.append(j)
                else:
                    drop.append(i)

    sources_indexes = [k for k, s in enumerate(sources_info) if k not in drop]
    sources_info = [s for k, s in enumerate(sources_info) if k not in drop]
    return sources_info, sources_indexes


def remove_outliers(reconstructed,
                    mu_ts,
                    FS=2222,
                    SENSITIVITY_STANDALONE=3,
                    MIN_STANDALONE=3):

    if len(mu_ts) > 3:

        fit = DBSCAN(eps=FS * SENSITIVITY_STANDALONE,
                     min_samples=MIN_STANDALONE,
                     leaf_size=1).fit_predict(mu_ts.reshape(-1, 1))

        mask = [s > -1 for s in fit]
        mu_ts = mu_ts[mask]

    return mu_ts


# @jit(nopython=True)
def gram_schmidt(w, B):
    projw_a = np.zeros_like(w)

    for i in range(B.shape[1]):
        a = B[:, i]

        # # Skip zero vectors
        # if np.linalg.norm(a) == 0:
        #     return w - projw_a

        dot_wa = np.dot(w, a)
        dot_aa = np.dot(a, a)
        projw_a += (dot_wa / dot_aa) * a

    return w - projw_a


# SKEW
# @jit(float64[:](float64[:]),nopython=True)
def CFdg_dw(w):
    return 3 * w**2


# @jit(float64[:](float64[:]),nopython=True)
def CFg(w):
    return w**3


def fastICA(signals,
            n_components,
            sil_thresh=0.8,
            pnr_thresh=10,
            cov_thresh=1.5,
            iterations_main=40,
            MIN_PEAKS_SRC=15,
            verbose=True,
            FS=2222,
            MIN_STANDALONE=3,
            SENSITIVITY_STANDALONE=3,
            DROP_ROA=0.8,
            cluster_method='k-means'):

    np.random.seed(7)

    global pca_object, agg_object

    pca_object = PCA(3, random_state=7)
    agg_object = AgglomerativeClustering(2,
                                         metric="euclidean",
                                         linkage="ward",
                                         compute_full_tree=False)

    # @jit(nopython=True)
    def func_cov(ISI):

        try:

            q1 = np.quantile(ISI, 0.25)
            q2, q3 = np.quantile(ISI, 0.5), np.quantile(ISI, 0.75)
            QCoV = (q3 - q1) / q2

        except:
            return 999.0

        return QCoV

    PEAK_DISTANCE = max(1, int(FS // 50))
    if verbose:
        print('min. peak distance=', PEAK_DISTANCE)

    import copy
    # detach
    signals = copy.deepcopy(signals)

    # sort ts by activity
    squared = np.abs(signals).mean(axis=0)

    PI = scipy_signal.find_peaks(squared, distance=PEAK_DISTANCE)[0]
    activity_index = np.argsort(squared[PI])[::-1]
    activity_index = PI[activity_index]

    n_components = min(n_components, len(activity_index) - 1)
    print('n_components:', n_components)

    # Initialize weights
    W = copy.deepcopy(signals[:, activity_index[:n_components]]).T

    source_indexes = []
    sources_info = []

    for c in range(n_components):

        if source_indexes and (c - source_indexes[-1]) > 100:
            print('>>Early stopping...')
            break

        if verbose:
            print(f'\nlooking for component: {c}')

        w = W[c, :]
        w = gram_schmidt(w, W[:c, :].T)
        #norms = np.linalg.norm(w)
        #w_new = w / norms

        for ii in range(iterations_main):

            ws = np.dot(w.T, signals)
            wg_ = CFdg_dw(ws)
            wg = CFg(ws).T

            # reconstructed - dv
            w_new = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()
            w_new = gram_schmidt(w_new, W[:c, :].T)

            norms = np.linalg.norm(w_new)
            w_new = w_new / norms

            lim = np.abs(np.abs((w_new * w).sum()) - 1.0)

            w = w_new
            W[c, :] = w.T

            if lim < 1e-8 and ii >= 11:
                print(f'reached limit at {ii+1}')
                break

        ################

        CoV = 999.0
        reconstructed = np.dot(w.T, signals)

        mu_ts, SIL0, SIL1, PNR0, PNR1 = get_mu_ts_sil_pnr(
            reconstructed, cluster_method, PEAK_DISTANCE, pca_object,
            agg_object)

        if len(mu_ts) > MIN_PEAKS_SRC:
            mu_ts = remove_outliers(
                reconstructed,
                mu_ts,
                FS=FS,
                MIN_STANDALONE=MIN_STANDALONE,
                SENSITIVITY_STANDALONE=SENSITIVITY_STANDALONE)

        mu_ts, SIL0, SIL1, PNR0, PNR1 = get_mu_ts_sil_pnr(
            reconstructed, cluster_method, PEAK_DISTANCE, pca_object,
            agg_object)

        if len(mu_ts) > MIN_PEAKS_SRC:
            ISI = np.diff(mu_ts)
            CoV = func_cov(ISI)

        condition = SIL1 >= sil_thresh and PNR1 >= pnr_thresh and CoV <= cov_thresh
        if len(mu_ts) >= MIN_PEAKS_SRC and condition:

            source_indexes.append(c)

            if verbose:
                print()
                print(f'found: {len(source_indexes)}; len: {len(mu_ts)}')
                print("LM {} SIL {:.2f} PNR {:.2f} QCoV {:.2f}".format(
                    len(mu_ts), SIL1, PNR1, CoV))
                print()

            sources_info.append({
                'reconstructed': reconstructed,
                'mu_ts': mu_ts,
                'SIL0': SIL0,
                'SIL1': SIL1,
                'PNR0': PNR0,
                'PNR1': PNR1,
                'IORDER': c
            })

        else:
            print("skipped: LM {} SIL {:.2f} PNR {:.2f} QCoV {:.2f}".format(
                len(mu_ts), SIL1, PNR1, CoV))

    sources = W[source_indexes, :]

    l1 = len(sources_info)
    sources_info, sources_indexes = drop_duplicates(sources_info, DROP_ROA)
    sources = sources[sources_indexes, :]
    l2 = len(sources_info)

    if l1 > 0:
        print(f"duplicates convergence ratio - {l1-l2}:{l1} = {(l1-l2)/l1}")
        DCR = (l1 - l2) / l1
    else:
        print(f"duplicates convergence ratio - NA, no MU identified")
        DCR = np.nan

    activity = np.argsort([len(s['mu_ts']) for s in sources_info])[::-1]
    sources_info = np.array(sources_info)[activity]
    sources = sources[activity, :]

    return sources, sources_info, DCR


def get_duration_mu(peaks, FS):
    try:
        return (np.max(peaks) - np.min(peaks)) / FS
    except Exception as e:
        print(e)
        return 1


def pps_per_sec_win(mu_ts, L, sec):
    PPS = []
    srange = np.zeros(L)
    srange[mu_ts] = 1

    PPS = moving_average(srange, sec) * sec

    return PPS


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def butterworth_filter(data, lowcut=20, highcut=500, fs=2222, order=2):

    highcut = min(int(fs / 2 - 1), highcut)
    b, a = scipy_signal.butter(order, [lowcut, highcut], fs=fs, btype="band")
    filtered_data = scipy_signal.lfilter(b, a, data)
    return filtered_data


def notch_filter(signal, fs=2222, notch_freq=50):
    quality_factor = 30.0  # idno
    b_notch, a_notch = scipy_signal.iirnotch(notch_freq, quality_factor, fs)
    signal = scipy_signal.filtfilt(b_notch, a_notch, signal)
    return signal


def wavelet_filter(signals, threshold=1, fs=2222):

    window = fs

    for i in range(signals.shape[0]):
        for j in range(0, signals.shape[1], window):
            signal = signals[i, j:j + window]
            w = pywt.Wavelet('db2')
            maxlev = pywt.dwt_max_level(len(signal), w.dec_len)
            coeffs = pywt.wavedec(signal, 'db2', level=maxlev)

            for ii in range(1, len(coeffs)):
                coeffs[ii] = pywt.threshold(coeffs[ii],
                                            threshold *
                                            np.quantile(coeffs[ii], 0.75),
                                            mode='soft')

            y = pywt.waverec(coeffs, 'db2')
            l = len(signals[i, j:j + window])
            signals[i, j:j + window] = y[:l]

    return signals


def get_colors(N, PALETTE):
    cmap = plt.cm.get_cmap(PALETTE, N)

    colors = [matplotlib_colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    return colors


import json


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


###########################################
def get_stats_sources_info_full(sources_info,
                                FORCE_CHANNEL,
                                rms,
                                n_time_points,
                                FS,
                                STAGE={}):

    stage_res = {}

    for mu, src in enumerate(sources_info):
        src['mu_orig_id'] = mu

        mu_ts = src['mu_ts']

        ISI = np.diff(mu_ts)

        src['avg. ISI'] = np.mean(ISI)
        src['sd. ISI'] = np.std(ISI)

        CoV = np.std(ISI) / np.mean(ISI)
        q1, q2, q3 = np.quantile(ISI, 0.25), np.quantile(ISI,
                                                         0.5), np.quantile(
                                                             ISI, 0.75)
        QCoV = (q3 - q1) / q2

        src['CoV'] = CoV
        src['QCoV'] = QCoV
        src["num. peaks"] = len(mu_ts)

        pps = pps_per_sec_win(mu_ts, n_time_points, FS)
        pps_avg = moving_average(pps, FS)
        AVGPPS = np.median([p for p in pps if p > 0])
        MAXPPS = np.quantile([p for p in pps if p > 0], 0.75)
        pps_avg = pps_avg / pps_avg.max() * MAXPPS

        pps1 = [p for p in pps if p > 0]

        src['init. pps'] = np.quantile(pps1, 0.25)
        src['term. pps'] = np.quantile(pps1, 0.75)

        src['avg. pps'] = AVGPPS
        src['max. pps'] = MAXPPS
        src['pps_avg'] = pps_avg

        ampl = np.zeros(n_time_points)
        ampl[mu_ts] = rms[mu_ts]
        ampl_avg = moving_average(ampl, FS)
        AVGAMPL = np.median([a for a in ampl if a > 0])
        MAXAMPL = np.quantile([a for a in ampl if a > 0], 0.75)
        ampl_avg = ampl_avg / ampl_avg.max() * MAXAMPL

        src['avg. ampl'] = AVGAMPL
        src['max. ampl'] = MAXAMPL
        src['ampl_avg'] = ampl_avg

        src["recruit. time"] = mu_ts[0] / FS
        src["derecruit. time"] = mu_ts[-1] / FS

        src["recruit. force"] = np.median(FORCE_CHANNEL[mu_ts[0]:mu_ts[0] + 3])
        src["derecruit. force"] = np.median(FORCE_CHANNEL[mu_ts[-1] -
                                                          3:mu_ts[-1]])

        for k, v in STAGE.items():
            try:
                li, ri = int(v['min']), int(v['max'])

                amplv = np.quantile([a for a in ampl[li:ri] if a > 0], 0.75)
                ppsv = np.median([a for a in pps[li:ri] if a > 0])

                stage_res[(f"MU{mu}", k)] = {
                    'avg. pps': ppsv,
                    'max. ampl': amplv
                }

            except:
                pass

    res = {}

    for mu, src in enumerate(sources_info):
        res[mu] = {
            k: src[k]
            for k in [
                'IORDER', 'PNR0', 'PNR1', 'SIL0', 'SIL1', 'CoV', 'QCoV',
                "num. peaks", 'avg. pps', 'avg. ampl', 'max. pps', 'max. ampl',
                "recruit. time", "derecruit. time", "recruit. force",
                "derecruit. force", 'avg. ISI', 'sd. ISI', 'init. pps',
                'term. pps'
            ]
        }

    res = pd.DataFrame(res).T
    stage_res = pd.DataFrame(stage_res).T if len(
        stage_res) > 0 else pd.DataFrame()
    return res, stage_res


############################################
def plot_regression(results_save_path, res):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    sns.despine()

    ax = axes[0]
    sns.regplot(x=res['max. ampl'],
                y=res['avg. pps'],
                ci=0,
                color='k',
                order=1,
                ax=ax)

    ax.set_xlabel('Max. ampl')
    ax.set_ylabel('Avg. pps')

    ax = axes[1]
    sns.regplot(x=res['IORDER'],
                y=res['avg. pps'],
                ci=0,
                color='k',
                order=1,
                ax=ax)

    ax.set_xlabel('Order of identification')
    ax.set_ylabel('Avg. pps')

    ax = axes[2]
    sns.regplot(x=res['IORDER'],
                y=res['max. ampl'],
                ci=0,
                color='k',
                order=1,
                ax=ax)

    ax.set_xlabel('Order of identification')
    ax.set_ylabel('Max. ampl')

    for ax in axes:
        ax.tick_params(pad=0)
        ax.set_ylabel(ax.get_ylabel(), labelpad=0)

    plt.savefig(f'{results_save_path}/regression_order1.pdf',
                dpi=300,
                bbox_inches='tight')

    # plt.close()


def plot_raster(results_save_path, FS, TS, SOURCES_INFO, FORCE_CHANNEL,
                cluster_method, extension_method):

    MU_FOUND = len(SOURCES_INFO)

    colors = get_colors(MU_FOUND, 'Spectral')

    fig, ax = plt.subplots(1,
                           1,
                           figsize=(8, MU_FOUND / 8),
                           width_ratios=[1],
                           sharey=True)

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.05)
    sns.despine(fig=fig)

    ax.margins(0)

    yticks = np.arange(MU_FOUND)
    ax.set_yticks(yticks + 0.5, labels=yticks + 1)

    ax.tick_params(pad=0)

    FC_MU = np.zeros((TS))
    for col in yticks:
        x = SOURCES_INFO[col]['mu_ts']
        FC_MU[x] += 1

        y = [col] * len(x)
        x = [[xs, xs, xs] for xs in x]
        x = [xx for xs in x for xx in xs]
        y = [[ys, ys + 1, ys] for ys in y]
        y = [yy for ys in y for yy in ys]
        x = [0] + x + [TS]
        y = [col] + y + [col]
        ax.plot(x, y, lw=0.3, c=colors[col], rasterized=False)

    ax.set_ylabel('MU')
    ax.set_xlabel('Time (s)')

    xticks = np.linspace(0, TS, TS)
    FC = FORCE_CHANNEL
    FC = moving_average(FC, FS)
    FC = (FC - FC.min()) / (FC.max() - FC.min())
    FC = FC * (MU_FOUND + 1)
    ax.plot(xticks, FC, lw=2, color='k')

    xticks = np.linspace(0, TS, 10)
    xlabels = np.linspace(0, TS / FS, 10).round().astype('int')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.text(1,
            1,
            cluster_method + '+' + extension_method,
            transform=ax.transAxes,
            color='#122726',
            va='bottom',
            ha='right')

    ax.spines['bottom'].set_color('#122726')
    ax.spines['top'].set_color('#122726')
    ax.spines['right'].set_color('#122726')
    ax.spines['left'].set_color('#122726')


    plt.savefig(f'{results_save_path}/raster.pdf',
                dpi=300,
                bbox_inches='tight')

    # plt.close()


def plot_ap_raw(results_save_path, CH, FS, MU_FOUND, DATA_FILTERED,
                SOURCES_INFO):
    W = int(FS / 100 * 7 * 2)
    if W % 2 == 1:
        W += 1

    scaler = 8.579150579150578
    scaler = int(FS / scaler)

    for MU in range(MU_FOUND):
        fig, axes = plt.subplots(1, CH, figsize=(CH, 1), sharey=True)

        for ch_row, ax in enumerate(axes):
            a, S, ind = get_mu_avg_shape_win(DATA_FILTERED[ch_row, :],
                                             SOURCES_INFO[MU]['mu_ts'],
                                             win=int(W * 1.8),
                                             recenter=False)

            sa = np.array([np.max(s) for s in S])
            si = np.argsort(sa)
            s = np.mean(np.array(S)[si][:], axis=0)

            #s = s * 10**6
            s[:scaler] = 0
            s[-scaler:] = 0

            ax.set_title(f"CH{ch_row+1}")
            ax.plot(s[100:-100], color='k', lw=1)

        axes[0].set_ylabel('MUAP')  #\n(muV)'')

        plt.savefig(f'{results_save_path}/AP/raw/avg_ap_raw_MU{MU+1}.png',
                    dpi=200,
                    bbox_inches='tight')
        # plt.close()


def plot_ap_smooth(results_save_path, CH, FS, MU_FOUND, DATA_FILTERED,
                   SOURCES_INFO):
    W = int(FS / 100 * 7 * 2)
    if W % 2 == 1:
        W += 1

    scaler = 8.579150579150578
    scaler = int(FS / scaler)

    smooth_factor = int(FS / 370.3333333333333)

    for MU in range(MU_FOUND):
        fig, axes = plt.subplots(1, CH, figsize=(CH, 1), sharey=True)

        for ch_row, ax in enumerate(axes):
            a, S, ind = get_mu_avg_shape_win(DATA_FILTERED[ch_row, :],
                                             SOURCES_INFO[MU]['mu_ts'],
                                             win=int(W * 1.8),
                                             recenter=False)

            sa = np.array([np.max(s) for s in S])
            si = np.argsort(sa)
            s = np.mean(np.array(S)[si][:], axis=0)

            #s = s * 10**6
            s[:scaler] = 0
            s[-scaler:] = 0

            s = gaussian_filter1d(s.astype(np.float64), smooth_factor)

            ax.set_title(f"CH{ch_row+1}")
            ax.plot(s[100:-100], color='k', lw=1)

        axes[0].set_ylabel('MUAP')  #\n(muV)')

        plt.savefig(
            f'{results_save_path}/AP/smooth/avg_ap_smooth_MU{MU+1}.png',
            dpi=200,
            bbox_inches='tight')
        # plt.close()


def deal_skew(df, x, l, r, step, low=False):

    vv = df[df[x] > r][x].values
    vv.sort()
    # print(x, ':', len(vv))
    for v in vv:
        df[x] = df[x].replace(v, r)
        r += step

    if low:
        vv = df[df[x] < l][x].values
        vv.sort()
        vv = vv[::-1]
        # print(x, ':', len(vv))
        for v in vv:
            df[x] = df[x].replace(v, l)
            l -= step

    return df


def skew_filter(signals, fs=2222, factor=1.5):

    window = fs

    for row in range(0, signals.shape[0], window):

        signal = signals.iloc[row:row + window]

        x = signal.values.flatten()
        step = np.quantile(np.abs(np.diff(x)), 0.05)

        q25 = np.nanquantile(x, 0.25)
        q75 = np.nanquantile(x, 0.75)
        iqr = q75 - q25

        l = q25 - iqr * factor
        r = q75 + iqr * factor

        for col in signal.columns:
            signal = deal_skew(signal, col, l, r, step, True)

        signals.loc[signal.index, signal.columns] = signal

    return signals


def drop_nan_align(data):
    ch_lens = []
    for col in list(data):
        l = len(data[col].dropna())
        ch_lens.append(l)

    ch_len = max(ch_lens)
    for col in list(data):
        data[col] = scipy_signal.resample(data[col].dropna(), ch_len)

    return data


def detach_force_channel_or_estimate_get_ams(data, FORCE_CHANNEL_FIRST=True, FS=2222):
    if not FORCE_CHANNEL_FIRST:
        # estimate force channel shape using mean abs amplitude
        FORCE_CHANNEL, AMS = data.abs().mean(axis=1).values
    else:
        FORCE_CHANNEL = data.iloc[:, 0].dropna().values
        data = data.iloc[:, 1:]
        AMS = data.abs().mean(axis=1).values

    FORCE_CHANNEL = moving_average(FORCE_CHANNEL, FS)
    AMS = moving_average(AMS, FS)

    FORCE_CHANNEL[FORCE_CHANNEL < 0] = 0

    force_min = np.quantile(FORCE_CHANNEL, 0.01)
    force_max = np.quantile(FORCE_CHANNEL, 0.99)

    FORCE_CHANNEL = ((FORCE_CHANNEL - force_min) /
                     (force_max - force_min)) * 100
    FORCE_CHANNEL[FORCE_CHANNEL < 0] = 0

    return data, FORCE_CHANNEL, AMS


def apply_bandpass_notch(data,
                        apply_butterworth_filter=True,
                        apply_notch_filter=True,
                        apply_wavelet_filter=False,
                        lowcut=30,
                        highcut=400,
                        FS=2222):

    SIGNAL = copy.deepcopy(data.T.values)
    CH, TS = SIGNAL.shape

    for ch_row in range(CH):

        s = moving_average(SIGNAL[ch_row, :], 3)
        if apply_butterworth_filter:
            s = butterworth_filter(s, lowcut, highcut, FS, order=2)
        if apply_notch_filter:

            N = len(s)
            yf = rfft(s)
            xf = rfftfreq(N, 1 / FS)

            mn = np.mean(np.abs(yf))
            sd = np.std(np.abs(yf))
            r = mn + sd * 10

            while np.max(np.abs(yf)) > r:
                N = len(s)
                yf = rfft(s)
                xf = rfftfreq(N, 1 / FS)
                notch_filter_hz = xf[np.argmax(np.abs(yf))]
                s = notch_filter(s, FS, notch_filter_hz)

            if apply_wavelet_filter:
                s = wavelet_filter(s.reshape(1, -1)).flatten()

        SIGNAL[ch_row, :] = s

    return SIGNAL


def get_extend(SIGNAL,
               apply_zscore=True,
               apply_extension=True,
               extension_method='spline',
               DEGREE=72):

    NUM_EXTEND = SPLINE_DEGREE = DEGREE

    if apply_zscore:
        scaler = StandardScaler()
        SIGNAL = scaler.fit_transform(SIGNAL.T).T

    if apply_extension:
        spline = None
        if extension_method == 'spline':

            spline = SplineTransformer(degree=SPLINE_DEGREE,
                                       n_knots=SPLINE_DEGREE + 1,
                                       extrapolation='periodic',
                                       knots='quantile',
                                       include_bias=False)

            SIGNAL = spline.fit_transform(SIGNAL.T).T

        if extension_method == 'extend':
            SIGNAL = extend(SIGNAL, NUM_EXTEND)

        keep = []
        for i in range(SIGNAL.shape[0]):
            if len(np.unique(SIGNAL[i, :])) < 10:
                continue
            keep.append(i)

        SIGNAL = SIGNAL[keep, :]

    return SIGNAL