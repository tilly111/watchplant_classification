import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, iqr
from scipy.signal import welch
import antropy as ant
from pywt import WaveletPacket
from tsfresh import extract_features

def wavelet_entropy(data, decomposition_level=4):
    # Calculate the wavelet packet entropy. Use a Daubechies 1 wavelet
    wp = WaveletPacket(data, 'db1')
    depth = wp.maxlevel
    wp.get_level(min(depth, decomposition_level))
    coeff_nodes = wp.get_leaf_nodes()
    # calculate the coefficients of WPT at given level
    coeffs = [node.data for node in coeff_nodes]
    entropy = 0
    # Run over all coefficients
    for wavelet in coeffs:
        for coeff in wavelet:
            if coeff != 0:
                # sum the entropy of all wavelet coefficients
                entropy -= (coeff**2)*(np.log(coeff**2))
    return entropy


def power_spectral_density(data):
    # Calculate PSD using welch estimate. PSD gives the spectra
    # depending on the frequencies. Sum over all spectra to
    # receive the total (average) spectral power
    _, psd = welch(data)  # TODO adjust parameters such as sampling frequency
    return sum(psd)


def calc_all_features(bg: pd.DataFrame, no: pd.DataFrame, stim: pd.DataFrame) -> list:
    """
    Calculates all 9 features for the given dataframes.
    Handover bg["differential_potential_pn1"], no["differential_potential_pn1"], stim["differential_potential_pn1"]
    :param bg: background data
    :param no: no stimulus data
    :param stim: stimulus data
    :return: list of all features
    """
    features = []

    # mean
    bg_mean = bg.mean(skipna=True)
    no_mean = no.mean(skipna=True)
    stim_mean = stim.mean(skipna=True)

    features.append(no_mean - bg_mean)
    features.append(stim_mean - bg_mean)

    # std
    bg_std = bg.std(skipna=True)
    no_std = no.std(skipna=True)
    stim_std = stim.std(skipna=True)

    features.append(no_std - bg_std)
    features.append(stim_std - bg_std)

    # variance
    bg_var = bg.var(skipna=True)
    no_var = no.var(skipna=True)
    stim_var = stim.var(skipna=True)

    features.append(no_var - bg_var)
    features.append(stim_var - bg_var)

    # make DataFrame to numpy array TODO is probably a hack to get rid of nans
    bg = (bg.interpolate(method="linear").fillna(value=0)).to_numpy()
    no = (no.interpolate(method="linear").fillna(value=0)).to_numpy()
    stim = (stim.interpolate(method="linear").fillna(value=0)).to_numpy()

    # print(bg.shape)
    # print(no.shape)
    # print(stim.shape)
    #
    # if np.isnan(bg).any():
    #     print("bg has nan values")
    # if np.isnan(no).any():
    #     print("no has nan values")
    # if np.isnan(stim).any():
    #     print("stim has nan values")

    # skewness
    bg_skew = skew(bg)
    no_skew = skew(no)
    stim_skew = skew(stim)

    features.append(no_skew - bg_skew)
    features.append(stim_skew - bg_skew)

    # kurtosis
    bg_kurt = kurtosis(bg)
    no_kurt = kurtosis(no)
    stim_kurt = kurtosis(stim)

    features.append(no_kurt - bg_kurt)
    features.append(stim_kurt - bg_kurt)

    # interquartile range
    bg_iqr = iqr(bg)
    no_iqr = iqr(no)
    stim_iqr = iqr(stim)

    features.append(no_iqr - bg_iqr)
    features.append(stim_iqr - bg_iqr)

    # hjorth mobility & hjorth complexity
    bg_hjorth_mob, bg_hjorth_comp = ant.hjorth_params(bg)
    no_hjorth_mob, no_hjorth_comp = ant.hjorth_params(no)
    stim_hjorth_mob, stim_hjorth_comp = ant.hjorth_params(stim)

    features.append(no_hjorth_mob - bg_hjorth_mob)
    features.append(stim_hjorth_mob - bg_hjorth_mob)
    features.append(no_hjorth_comp - bg_hjorth_comp)
    features.append(stim_hjorth_comp - bg_hjorth_comp)

    # wavelet packet entropy
    bg_wavelet_entropy = wavelet_entropy(bg)
    no_wavelet_entropy = wavelet_entropy(no)
    stim_wavelet_entropy = wavelet_entropy(stim)

    features.append(no_wavelet_entropy - bg_wavelet_entropy)
    features.append(stim_wavelet_entropy - bg_wavelet_entropy)

    # average spectral power
    bg_power_spectral_density = power_spectral_density(bg)
    no_power_spectral_density = power_spectral_density(no)
    stim_power_spectral_density = power_spectral_density(stim)

    features.append(no_power_spectral_density - bg_power_spectral_density)
    features.append(stim_power_spectral_density - bg_power_spectral_density)

    return features


def calc_ts_fresh(bg: pd.DataFrame, no: pd.DataFrame, stim: pd.DataFrame) -> pd.DataFrame:
    if not bg.empty:
        bg.dropna(axis=0, inplace=True)
        bg_df = bg.to_frame()
        bg_df["id"] = np.ones((bg_df.shape[0],))
        f_bg = extract_features(bg_df, column_id='id', column_value=bg_df.columns[0], n_jobs=1)
    else:
        f_bg = None
    if not no.empty:
        no.dropna(axis=0, inplace=True)
        no_df = no.to_frame()
        no_df["id"] = np.ones((no_df.shape[0],))
        f_no = extract_features(no_df, column_id='id', column_value=no_df.columns[0], n_jobs=1)
    else:
        f_no = None
    if not stim.empty:
        stim.dropna(axis=0, inplace=True)
        stim_df = stim.to_frame()
        stim_df["id"] = np.ones((stim_df.shape[0],))
        f_stim = extract_features(stim_df, column_id='id', column_value=stim_df.columns[0], n_jobs=1)
    else:
        f_stim = None

    if f_bg is not None and f_no is not None and f_stim is not None:
        f_stim = f_stim - f_bg
        f_no = f_no - f_bg

        res = pd.concat([f_stim, f_no], axis=1)
        print(f"type of return:{type(res)}")
        if type(res) is pd.DataFrame:
            return res
        elif type(res) is pd.Series:
            return res.to_frame()
        else:
            print("Some weird error. Returning None.")
            return None
    else:
        print("Series incomplete. Returning empty list.")
        return None
