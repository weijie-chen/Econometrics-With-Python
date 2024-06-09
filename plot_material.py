import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import scipy as sp
import scipy.stats
import statsmodels.api as sm


###############################################################################################################
###############################################################################################################
def reg_corr_plot():
    class LinearRegression:
        def __init__(self, beta1, beta2, error_scale, data_size):
            self.beta1 = beta1
            self.beta2 = beta2
            self.error_scale = error_scale
            self.x = np.random.randint(1, data_size, data_size)
            self.y = (
                self.beta1
                + self.beta2 * self.x
                + self.error_scale * np.random.randn(data_size)
            )

        def x_y_cor(self):
            return np.corrcoef(self.x, self.y)[0, 1]

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 12))

    beta1, beta2, error_scale, data_size = 2, 0.05, 1, 100
    lrg1 = LinearRegression(beta1, beta2, error_scale, data_size)
    ax[0, 0].scatter(lrg1.x, lrg1.y)
    ax[0, 0].plot(lrg1.x, beta1 + beta2 * lrg1.x, color="#FA954D", alpha=0.7)
    ax[0, 0].set_title(r"$Y={}+{}X+{}u$".format(beta1, beta2, error_scale))
    ax[0, 0].annotate(
        r"$\rho={:.4}$".format(lrg1.x_y_cor()), xy=(0.1, 0.9), xycoords="axes fraction"
    )

    beta1, beta2, error_scale, data_size = 2, -0.6, 1, 100
    lrg2 = LinearRegression(beta1, beta2, error_scale, data_size)
    ax[0, 1].scatter(lrg2.x, lrg2.y)
    ax[0, 1].plot(lrg2.x, 2 - 0.6 * lrg2.x, color="#FA954D", alpha=0.7)
    ax[0, 1].set_title(r"$Y={}+{}X+{}u$".format(beta1, beta2, error_scale))
    ax[0, 1].annotate(
        r"$\rho={:.4}$".format(lrg2.x_y_cor()), xy=(0.1, 0.9), xycoords="axes fraction"
    )

    beta1, beta2, error_scale, data_size = 2, 1, 1, 100
    lrg3 = LinearRegression(beta1, beta2, error_scale, data_size)
    ax[0, 2].scatter(lrg3.x, lrg3.y)
    ax[0, 2].plot(lrg3.x, beta1 + beta2 * lrg3.x, color="#FA954D", alpha=0.7)
    ax[0, 2].set_title(r"$Y={}+{}X+{}u$".format(beta1, beta2, error_scale))
    ax[0, 2].annotate(
        r"$\rho={:.4}$".format(lrg3.x_y_cor()), xy=(0.1, 0.9), xycoords="axes fraction"
    )

    beta1, beta2, error_scale, data_size = 2, 3, 1, 100
    lrg4 = LinearRegression(beta1, beta2, error_scale, data_size)
    ax[0, 3].scatter(lrg4.x, lrg4.y)
    ax[0, 3].plot(lrg4.x, beta1 + beta2 * lrg4.x, color="#FA954D", alpha=0.7)
    ax[0, 3].set_title(r"$Y={}+{}X+{}u$".format(beta1, beta2, error_scale))
    ax[0, 3].annotate(
        r"$\rho={:.4}$".format(lrg4.x_y_cor()), xy=(0.1, 0.9), xycoords="axes fraction"
    )

    beta1, beta2, error_scale, data_size = 2, 3, 3, 100
    lrg5 = LinearRegression(beta1, beta2, error_scale, data_size)
    ax[1, 0].scatter(lrg5.x, lrg5.y)
    ax[1, 0].plot(lrg5.x, beta1 + beta2 * lrg5.x, color="#FA954D", alpha=0.7)
    ax[1, 0].set_title(r"$Y={}+{}X+{}u$".format(beta1, beta2, error_scale))
    ax[1, 0].annotate(
        r"$\rho={:.4}$".format(lrg5.x_y_cor()), xy=(0.1, 0.9), xycoords="axes fraction"
    )

    beta1, beta2, error_scale, data_size = 2, 3, 10, 100
    lrg6 = LinearRegression(beta1, beta2, error_scale, data_size)
    ax[1, 1].scatter(lrg6.x, lrg6.y)
    ax[1, 1].plot(lrg6.x, beta1 + beta2 * lrg6.x, color="#FA954D", alpha=0.7)
    ax[1, 1].set_title(r"$Y={}+{}X+{}u$".format(beta1, beta2, error_scale))
    ax[1, 1].annotate(
        r"$\rho={:.4}$".format(lrg6.x_y_cor()), xy=(0.1, 0.9), xycoords="axes fraction"
    )

    beta1, beta2, error_scale, data_size = 2, 3, 20, 100
    lrg7 = LinearRegression(beta1, beta2, error_scale, data_size)
    ax[1, 2].scatter(lrg7.x, lrg7.y)
    ax[1, 2].plot(lrg7.x, beta1 + beta2 * lrg7.x, color="#FA954D", alpha=0.7)
    ax[1, 2].set_title(r"$Y={}+{}X+{}u$".format(beta1, beta2, error_scale))
    ax[1, 2].annotate(
        r"$\rho={:.4}$".format(lrg7.x_y_cor()), xy=(0.1, 0.9), xycoords="axes fraction"
    )

    beta1, beta2, error_scale, data_size = 2, 3, 50, 100
    lrg8 = LinearRegression(beta1, beta2, error_scale, data_size)
    ax[1, 3].scatter(lrg8.x, lrg8.y)
    ax[1, 3].plot(lrg3.x, beta1 + beta2 * lrg3.x, color="#FA954D", alpha=0.7)
    ax[1, 3].set_title(r"$Y={}+{}X+{}u$".format(beta1, beta2, error_scale))
    ax[1, 3].annotate(
        r"$\rho={:.4}$".format(lrg8.x_y_cor()), xy=(0.1, 0.9), xycoords="axes fraction"
    )


###############################################################################################################
###############################################################################################################


def central_limit_theorem_plot():
    fig, ax = plt.subplots(4, 3, figsize=(20, 20))

    ########################################################################################
    x = np.linspace(2, 8, 100)
    a = 2  # range of uniform distribution
    b = 8
    unif_pdf = np.ones(len(x)) * 1 / (b - a)

    ax[0, 0].plot(x, unif_pdf, lw=3, color="r")
    ax[0, 0].plot(
        [x[0], x[0]], [0, 1 / (b - a)], lw=3, color="r", alpha=0.9
    )  # vertical line
    ax[0, 0].plot([x[-1], x[-1]], [0, 1 / (b - a)], lw=3, color="r", alpha=0.9)
    ax[0, 0].fill_between(x, 1 / (b - a), 0, alpha=0.5, color="r")

    ax[0, 0].set_xlim([1, 9])
    ax[0, 0].set_ylim([0, 0.4])
    ax[0, 0].set_title("Uniform Distribution", size=18)
    ax[0, 0].set_ylabel("Population Distribution", size=12)

    ########################################################################################
    ss = 2  # sample size
    unif_sample_mean = np.zeros(1000)
    for i in range(1000):
        unif_sample = np.random.rand(ss)
        unif_sample_mean[i] = np.mean(unif_sample)
    ax[1, 0].hist(unif_sample_mean, bins=20, color="r", alpha=0.5)
    ax[1, 0].set_ylabel("Sample Distribution， $n = 2$", size=12)
    ########################################################################################
    ss = 10  # sample size
    unif_sample_mean = np.zeros(1000)
    for i in range(1000):
        unif_sample = np.random.rand(ss)
        unif_sample_mean[i] = np.mean(unif_sample)
    ax[2, 0].hist(unif_sample_mean, bins=30, color="r", alpha=0.5)
    ax[2, 0].set_ylabel("Sample Distribution, $n = 10$", size=12)

    ########################################################################################
    ss = 1000  # sample size
    unif_sample_mean = np.zeros(1000)
    for i in range(1000):
        unif_sample = np.random.rand(ss)
        unif_sample_mean[i] = np.mean(unif_sample)
    ax[3, 0].hist(unif_sample_mean, bins=40, color="r", alpha=0.5)
    ax[3, 0].set_ylabel("Sample Distribution, $n = 1000$", size=12)

    ########################################################################################
    a = 6
    b = 2
    x = np.linspace(0, 1, 100)
    beta_pdf = sp.stats.beta.pdf(x, a, b)
    ax[0, 1].plot(x, beta_pdf, lw=3, color="g")
    ax[0, 1].set_ylim([0, 6])
    ax[0, 1].fill_between(x, beta_pdf, 0, alpha=0.5, color="g")
    ax[0, 1].set_title("Beta Distribution", size=18)
    ########################################################################################

    ss = 2  # sample size
    beta_sample_mean = np.zeros(1000)
    for i in range(1000):
        beta_sample = sp.stats.beta.rvs(a, b, size=ss)
        beta_sample_mean[i] = np.mean(beta_sample)
    ax[1, 1].hist(beta_sample_mean, color="g", alpha=0.5)

    ########################################################################################

    ss = 10  # sample size
    beta_sample_mean = np.zeros(1000)
    for i in range(1000):
        beta_sample = sp.stats.beta.rvs(a, b, size=ss)
        beta_sample_mean[i] = np.mean(beta_sample)
    ax[2, 1].hist(beta_sample_mean, color="g", bins=20, alpha=0.5)

    ########################################################################################

    ss = 100000  # sample size
    beta_sample_mean = np.zeros(1000)
    for i in range(1000):
        beta_sample = sp.stats.beta.rvs(a, b, size=ss)
        beta_sample_mean[i] = np.mean(beta_sample)
    ax[3, 1].hist(beta_sample_mean, color="g", bins=30, alpha=0.5)

    ########################################################################################
    a = 6
    x = np.linspace(0, 25, 100)

    gamma_pdf = sp.stats.gamma.pdf(x, a)
    ax[0, 2].plot(x, gamma_pdf, lw=3, color="b")
    ax[0, 2].set_ylim([0, 0.34])
    ax[0, 2].fill_between(x, gamma_pdf, 0, alpha=0.5, color="b")
    ax[0, 2].set_title("Gamma Distribution", size=18)

    ########################################################################################
    ss = 2  # sample size
    gamma_sample_mean = np.zeros(1000)
    for i in range(1000):
        gamma_sample = sp.stats.gamma.rvs(a, size=ss)
        gamma_sample_mean[i] = np.mean(gamma_sample)
    ax[1, 2].hist(gamma_sample_mean, color="b", alpha=0.5)

    ########################################################################################
    ss = 10  # sample size
    gamma_sample_mean = np.zeros(1000)
    for i in range(1000):
        gamma_sample = sp.stats.gamma.rvs(a, size=ss)
        gamma_sample_mean[i] = np.mean(gamma_sample)
    ax[2, 2].hist(gamma_sample_mean, bins=20, color="b", alpha=0.5)
    ########################################################################################
    ss = 1000  # sample size
    gamma_sample_mean = np.zeros(1000)
    for i in range(1000):
        gamma_sample = sp.stats.gamma.rvs(a, size=ss)
        gamma_sample_mean[i] = np.mean(gamma_sample)
    ax[3, 2].hist(gamma_sample_mean, bins=30, color="b", alpha=0.5)
    ########################################################################################
    plt.show()


##########################################################################################################
##########################################################################################################
def type12_error():
    x = np.linspace(-6, 9, 200)
    null_loc, alter_loc = 0, 3
    y_null = sp.stats.norm.pdf(x, loc=null_loc)
    y_alter = sp.stats.norm.pdf(x, loc=alter_loc)
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(x, y_null, x, y_alter)
    ax.annotate("Null", (null_loc - 0.2, max(y_null) / 2), size=15)
    ax.annotate("Alternative", (alter_loc - 0.6, max(y_alter) / 2), size=15)
    ax.annotate("Type I Error", (2, max(y_alter) / 30), size=15)
    ax.annotate("Type II Error", (0, max(y_alter) / 30), size=15)
    ax.fill_between(x[-98:], y_null[-98:])
    ax.fill_between(x[:103], y_alter[:103])
    ax.set_ylim([0, 0.5])
    plt.show()


##########################################################################################################
##########################################################################################################
# def reject_region():
#     data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')

#     male_mean = data[data['Gender']=='Male']['Height'].mean()
#     male_std = data[data['Gender']=='Male']['Height'].std(ddof=1)
#     male_std_error = male_std/np.sqrt(len(data[data['Gender']=='Male']))
#     male_null = 172

#     df = len(data[data['Gender']=='Male'])-1
#     t_975 = sp.stats.t.ppf(.975, df=df)
#     t_025 = sp.stats.t.ppf(.025, df=df)

#     x = np.linspace(male_null-5, male_null+5, 200)
#     df = len(data[data['Gender']=='Male'])-1
#     y_t = sp.stats.t.pdf(x, df = df, loc = male_null)

#     fig, ax = plt.subplots(2, 1, figsize = (18,8))

#     ax[0].plot(x, y_t, color = 'tomato', lw = 3)

#     rejection_lower = male_null - t_975*male_std_error
#     x_rej_lower = np.linspace(rejection_lower-3, rejection_lower, 30)
#     y_rej_lower = sp.stats.t.pdf(x_rej_lower, df = df, loc = male_null)
#     ax[0].fill_between(x_rej_lower, y_rej_lower, color = 'tomato', alpha = .7)

#     rejection_upper = male_null + t_975*male_std_error
#     x_rej_upper = np.linspace(rejection_upper, rejection_upper+3, 30)
#     y_rej_upper = sp.stats.t.pdf(x_rej_upper, df = df, loc = male_null)
#     ax[0].fill_between(x_rej_upper, y_rej_upper, color = 'tomato', alpha = .7)

#     ax[0].set_ylim([0, .45])

#     x = np.linspace(-5, 5, 200)
#     y_t = sp.stats.t.pdf(x, df = df, loc = 0)

#     ax[1].plot(x, y_t, color = 'tomato', lw = 3)

#     x_rej_lower = np.linspace(t_025-3, t_025, 30)
#     y_rej_lower = sp.stats.t.pdf(x_rej_lower, df = df)
#     ax[1].fill_between(x_rej_lower, y_rej_lower, color = 'tomato', alpha = .7)

#     x_rej_lower = np.linspace(t_975+3, t_975, 30)
#     y_rej_lower = sp.stats.t.pdf(x_rej_lower, df = df)
#     ax[1].fill_between(x_rej_lower, y_rej_lower, color = 'tomato', alpha = .7)

#     ax[1].set_ylim([0, .45])

#     plt.show()

##########################################################################################################
# ##########################################################################################################
# def draw_something():
#     x = np.linspace(0, 10, 100)
#     y = np.sin(x)
#     plt.plot(x, y)


##########################################################################################################
##########################################################################################################
def anova_plot():
    def gen_3samples(loc1, loc2, loc3, scale1, scale2, scale3, size1, size2, size3):
        F_statistic, p_value = [], []
        for i in range(1000):
            a = sp.stats.norm.rvs(loc1, scale1, size1)
            b = sp.stats.norm.rvs(loc2, scale2, size3)
            c = sp.stats.norm.rvs(loc3, scale3, size3)
            F, p = sp.stats.f_oneway(a, b, c)
            F_statistic.append(F)
            p_value.append(p)
        return F_statistic, p_value

    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(17, 34))

    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        6,
        9,
        6,
        6,
        6,
        10,
        20,
        30,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[0, 0].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[0, 0].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[0, 0].set_title("Simulation 1")
    ax[0, 0].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[1, 0].hist(p_value, bins=50)

    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        3.1,
        2.9,
        6,
        6,
        6,
        10,
        20,
        30,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[0, 1].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[0, 1].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[0, 1].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[1, 1].hist(p_value, bins=50)
    ax[0, 1].set_title("Simulation 2")

    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        3.1,
        2.9,
        6,
        12,
        18,
        10,
        20,
        30,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[0, 2].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[0, 2].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[0, 2].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[1, 2].hist(p_value, bins=50)
    ax[0, 2].set_title("Simulation 3")

    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        6,
        9,
        10,
        10,
        10,
        10,
        20,
        30,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[2, 0].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[2, 0].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[2, 0].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[3, 0].hist(p_value, bins=50)
    ax[2, 0].set_title("Simulation 4")

    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        5,
        6,
        10,
        10,
        10,
        10,
        10,
        10,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[2, 1].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[2, 1].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[2, 1].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[3, 1].hist(p_value, bins=50)
    ax[2, 1].set_title("Simulation 5")

    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        5,
        6,
        10,
        10,
        10,
        5000,
        5000,
        5000,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[2, 2].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[2, 2].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[2, 2].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[3, 2].hist(p_value, bins=50)
    ax[2, 2].set_title("Simulation 6")

    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        3,
        3,
        100,
        100,
        100,
        10,
        10,
        10,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[4, 0].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[4, 0].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[4, 0].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[5, 0].hist(p_value, bins=50)
    ax[4, 0].set_title("Simulation 7")

    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        3,
        3,
        1,
        1,
        2,
        10,
        20,
        30,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[4, 1].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[4, 1].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[4, 1].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[5, 1].hist(p_value, bins=50)
    ax[4, 1].set_title("Simulation 8")

    params = [3, 3.1, 2.9, 0.01, 0.01, 0.01, 10, 20, 30]
    mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3 = [
        3,
        3,
        3,
        1,
        1,
        2,
        10,
        20,
        30,
    ]
    params = [mu1, mu2, mu3, sig1, sig2, sig3, size1, size2, size3]
    F_statistic, p_value = gen_3samples(*params)
    n, bins, patches = ax[4, 2].hist(F_statistic, bins=50)
    F_critical = sp.stats.f.ppf(0.95, 2, size1 + size2 + size3 - 3)
    textstr = "\n".join(
        (
            "$\mu_1, \mu_2, \mu_3 = {}, {}, {}$".format(mu1, mu2, mu3),
            "$\sigma_1, \sigma_2, \sigma_3 = {}, {}, {}$".format(sig1, sig2, sig3),
            "$n_1, n_2, n_3 = {}, {}, {}$".format(size1, size2, size3),
            r"$F_c = {:.4f}$".format(F_critical),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax[4, 2].text(
        max(bins) / 2,
        max(n) / 2,
        textstr,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax[4, 2].vlines(F_critical, 0, max(n) * 1.1, color="r")
    ax[5, 2].hist(p_value, bins=50)
    ax[4, 2].set_title("Simulation 9")

    #######################||Rectangle||##########################
    ##############################################################

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.10, 0.633),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.3755, 0.633),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.650, 0.633),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.10, 0.37),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.3755, 0.37),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.650, 0.37),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.650, 0.108),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.3755, 0.108),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.1, 0.108),
        0.2645,
        0.258,
        fill=False,
        color="k",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])
    ####################################################################

    plt.show()


###############################################################################################
###############################################################################################
def two_tail_rej_region_demo():
    data = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")

    df = len(data[data["Gender"] == "Male"]) - 1
    t_975 = sp.stats.t.ppf(0.975, df=df)
    t_025 = sp.stats.t.ppf(0.025, df=df)

    male_mean = data[data["Gender"] == "Male"]["Height"].mean()
    male_std = data[data["Gender"] == "Male"]["Height"].std(ddof=1)
    male_std_error = male_std / np.sqrt(len(data[data["Gender"] == "Male"]))
    male_null = 172

    x = np.linspace(male_null - 5, male_null + 5, 200)

    y_t = sp.stats.t.pdf(x, df=df, loc=male_null)

    fig, ax = plt.subplots(2, 1, figsize=(18, 8))

    ax[0].plot(x, y_t, color="tomato", lw=3)

    rejection_lower = male_null - t_975 * male_std_error
    x_rej_lower = np.linspace(rejection_lower - 3, rejection_lower, 30)
    y_rej_lower = sp.stats.t.pdf(x_rej_lower, df=df, loc=male_null)
    ax[0].fill_between(x_rej_lower, y_rej_lower, color="tomato", alpha=0.7)

    rejection_upper = male_null + t_975 * male_std_error
    x_rej_upper = np.linspace(rejection_upper, rejection_upper + 3, 30)
    y_rej_upper = sp.stats.t.pdf(x_rej_upper, df=df, loc=male_null)
    ax[0].fill_between(x_rej_upper, y_rej_upper, color="tomato", alpha=0.7)
    ax[0].set_ylim([0, 0.45])
    ax[0].set_title("Rejection Region of Original Unit (cm)")

    x = np.linspace(-5, 5, 200)
    y_t = sp.stats.t.pdf(x, df=df, loc=0)

    ax[1].plot(x, y_t, color="tomato", lw=3)

    x_rej_lower = np.linspace(t_025 - 3, t_025, 30)
    y_rej_lower = sp.stats.t.pdf(x_rej_lower, df=df)
    ax[1].fill_between(x_rej_lower, y_rej_lower, color="tomato", alpha=0.7)

    x_rej_lower = np.linspace(t_975 + 3, t_975, 30)
    y_rej_lower = sp.stats.t.pdf(x_rej_lower, df=df)
    ax[1].fill_between(x_rej_lower, y_rej_lower, color="tomato", alpha=0.7)
    ax[1].set_ylim([0, 0.45])
    ax[1].set_title("Rejection Region of t-statistic")

    plt.show()


###############################################################################################
###############################################################################################
def one_tail_rej_region_demo():
    fig, ax = plt.subplots(2, 1, figsize=(18, 8))
    x = np.linspace(-5, 5, 200)
    y_t = sp.stats.t.pdf(x, df=len(x), loc=0)
    ax[0].plot(x, y_t, color="tomato")

    ax[0].annotate("$H_0: \mu = \mu_0$\n$H_1: \mu < \mu_0$", (-5, 0.35), size=16)
    t_05 = sp.stats.t.ppf(0.05, df=len(x))
    x_rej_lower = np.linspace(t_05, t_05 - 3, 30)
    y_rej_lower = sp.stats.t.pdf(x_rej_lower, df=len(x))
    ax[0].fill_between(x_rej_lower, y_rej_lower, color="tomato", alpha=0.7)
    ax[0].set_ylim([0, 0.45])

    x = np.linspace(-5, 5, 200)
    y_t = sp.stats.t.pdf(x, df=len(x), loc=0)
    ax[1].plot(x, y_t, color="tomato")

    ax[1].annotate("$H_0: \mu = \mu_0$\n$H_1: \mu > \mu_0$", (4, 0.35), size=16)
    t_95 = sp.stats.t.ppf(0.95, df=len(x))
    x_rej_lower = np.linspace(t_95, t_95 + 3, 30)
    y_rej_lower = sp.stats.t.pdf(x_rej_lower, df=len(x))
    ax[1].fill_between(x_rej_lower, y_rej_lower, color="tomato", alpha=0.7)
    ax[1].set_ylim([0, 0.45])

    plt.show()


###############################################################################################
###############################################################################################


class Data:

    def __init__(self, **kwargs):
        """Input start, end, database"""
        self.__dict__.update(kwargs)
        self.start = start
        self.end = end
        self.database = database

    def retrieve(self, data_id):
        if self.database == "fred":
            self.df = pdr.data.DataReader(data_id, self.database, self.start, self.end)
        elif self.database == "oecd":
            self.df = pdr.data.DataReader(data_id, self.database)
        elif self.database == "eurostat":
            self.df = pdr.data.DataReader(data_id, self.database)

    def normalise(self):
        self.df_normalised = self.df / self.df.iloc[1]

    def plot(self, labels, grid_on, norm):
        if norm == False:
            self.labels = labels
            self.grid_on = grid_on

            fig, ax = plt.subplots(figsize=(14, 8))
            for col, label in zip(
                self.df, self.labels
            ):  # for drawing multiple labels/legends
                ax.plot(self.df_normalised[col], label=label)
            ax.grid(grid_on)
            ax.legend()
            plt.show()
        else:
            self.label = labels
            self.grid_on = grid_on

            fig, ax = plt.subplots(figsize=(14, 8))
            for col, label in zip(self.df_normalised, self.label):
                ax.plot(self.df_normalised[col], label=label)
            ax.legend()
            ax.grid(grid_on)
            plt.show()

    def twin_plot(
        self, lhs, rhs, labels, grid_on, ax_rhs_inverse, lhs_color, rhs_color
    ):
        self.lhs = lhs
        self.rhs = rhs
        self.labels = labels
        self.grid_on = grid_on
        self.ax_rhs_inverse = ax_rhs_inverse
        self.lhs_color = lhs_color
        self.rhs_color = rhs_color

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(self.df[self.lhs].dropna(), label=labels[0], color=self.lhs_color)
        ax.legend(loc=3)
        ax_RHS = ax.twinx()  # share the same x-axis
        ax_RHS.plot(self.df[self.rhs].dropna(), label=labels[1], color=self.rhs_color)
        ax_RHS.legend(loc=0)
        if ax_rhs_inverse == True:
            ax_RHS.invert_yaxis()
        ax.grid(grid_on)
        plt.show()
