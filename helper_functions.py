import scipy.stats as _stats
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

def choose_grid(n_features):
    NR_COLUMNS = 3
    if n_features < NR_COLUMNS:
        return 1, n_features
    else:
        return (n_features // NR_COLUMNS, NR_COLUMNS) if n_features % NR_COLUMNS == 0 else (n_features // NR_COLUMNS + 1, NR_COLUMNS)

def set_axes(xvalues: list, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '', percentage=False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xticklabels(xvalues, fontsize=8, ha='center')

    return ax

def save(dirname: str, chartname: str, index: str = ''):
    plt.savefig(dirname + chartname + index)
    

##### Scatter plots #####
def multiple_scatter_plots(data: pd.DataFrame):
    rows, cols = len(data.columns)-1, len(data.columns)-1
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4), squeeze=False)
    
    plt.subplots_adjust(wspace=0.9,hspace=0.9)
    
    columns = data.columns
    for i in range(len(data.columns)):
        var1 = data[columns[i]]
        for j in range(i+1,len(data.columns)):
            var2 = data[columns[j]]
            axs[i,j-1].set_title(f"{columns[i]} x {columns[j]}")
            axs[i,j-1].scatter(var1,var2)
            axs[i,j-1].set_xlabel(columns[i])
            axs[i,j-1].set_ylabel(columns[j])
            
    
    
##### Distributions plots #####

def multiple_line_chart(xvalues: list, yvalues: dict, hist_values: tuple, ax: plt.Axes = None, title: str = '',
                        xlabel: str = '', ylabel: str = '', percentage=False):
    ax = set_axes(xvalues, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    ax = set_locators(xvalues, hist_values, ax=ax)

    legend: list = []
    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend)
    plt.tight_layout()



def iterate_through_data(data: pd.DataFrame, features_names: pd.Index, rows: int, cols: int):
    height = 4
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * height, rows * height))
    i, j = 0, 0
    n = 0
    
    print(f'Rows {rows}, Columns {cols}')
    for feature in features_names:
        print(f'Feature {feature}')
        if rows == 1:
            if len(data[feature].unique()) > 2:
                histogram_with_distributions(axs[i], data[feature].dropna(), feature)
            else:
                boolean_histogram(axs[i], data[feature].dropna(), feature)
            i+=1
        else:
            if len(data[feature].unique()) > 2:
                histogram_with_distributions(axs[i, j], data[feature].dropna(), feature)
            else:
                boolean_histogram(axs[i, j], data[feature].dropna(), feature)
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
            n+=1
    save('charts/eda_charts/','numeric_distributions')
    
def set_locators(xvalues: list, hist_values: tuple, ax: plt.Axes = None):
    if isinstance(xvalues[0], dt.datetime):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator, defaultfmt='%Y-%m-%d'))
    else:
        #ax.set_xticks(xvalues)
        #ax.set_xlim(xvalues[0], xvalues[-1])
        # this median is for distributions that present a very extreme and very different probability
        # compared to the others
        ax.set_ylim(0, np.max(hist_values[0])+((np.max(hist_values[0])-np.median(hist_values[0])))/2)
        #pass

    return ax
    
def compute_known_distributions(x_values: list) -> dict:
    distributions = {}
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f, %.2f)' % (mean, sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # Exponencial
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)' % (1 / scale)] = _stats.expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = _stats.lognorm.fit(x_values)
    distributions['LogNor(%.1f, %.2f)' % (np.log(scale), sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    return distributions


def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    hist_values = ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, hist_values, ax=ax, title='Best fit for %s' % var)


def boolean_histogram(ax: plt.Axes, series: pd.Series, count=False):
    if not count:
        values = series.sort_values().values
    else:
        values = series.value_counts().index.to_list()
    sns.histplot(values, stat="density", ax=ax)

def identify_best_fit_distribution(df, feature):

    size = df[feature].shape[0]
    dist_names = ['norm','expon','lognorm']

    chi_square_statistics = []
    # 11 equi-distant bins of observed Data 
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(df[feature], percentile_bins)
    observed_frequency, bins = (np.histogram(df[feature], bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(_stats, distribution)
        param = dist.fit(df[feature])
        print("{}\n{}\n".format(dist, param))


        # Get expected counts in percentile bins
        # cdf of fitted distribution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square_statistics.append(ss)


    #Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True, ignore_index = True)


    print ('\nDistributions listed by Betterment of fit:')
    print ('............................................')
    print (results)

    return results['Distribution'][0]
