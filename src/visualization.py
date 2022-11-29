import numpy
import matplotlib.pyplot as plt

# compute the median of each column
def med(data):
    median = numpy.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = numpy.median(data[:, i])
    return median

def perc(data):
    # Quartiles are computed with numpy in the same way as the median, but using the function percentile.
    median = numpy.zeros(data.shape[1])
    perc_25 = numpy.zeros(data.shape[1])
    perc_75 = numpy.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = numpy.median(data[:, i])
        perc_25[i] = numpy.percentile(data[:, i], 25)
        perc_75[i] = numpy.percentile(data[:, i], 75)
    return median, perc_25, perc_75

def plot_convergence_plot(metric, xlabel = '', ylabel = '', title = '', figsize = (12,8), fontsize = 15, linewidth = 3, colorcode = '#006BB2', saveplot = False, savingname = 'MetricPlot'):
    
    plt.figure(figsize = figsize)
    plt.plot(metric, linewidth = linewidth, color = colorcode)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.title(title, fontsize = fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid()
    plt.draw()
    if saveplot:
        plt.savefig(savingname + '.pdf', format  ='pdf', dpi = 1500)


def plot_convergence_plot_for_multiple_run(metric, xlabel = '', ylabel = '', figsize = (12,8), fontsize = 15, 
                                           linewidth = 3, colorcode = '#006BB2', saveplot = False, savingname = 'MetricPlot'):
    """
    metric   : numpy.ndarray for metric. Shape : (n_trials, n_epochs)
    """

    last_results = metric[:,-1]
    title = 'Number of Trial : {}, \nBest Result : {:.3f}, Worst Result : {:.3f}, \nMean : {:.3f}, Median : {:.3f}'.format(last_results.shape[0],numpy.max(last_results), numpy.min(last_results), numpy.mean(last_results), numpy.median(last_results))
    params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': fontsize,
   'xtick.labelsize': fontsize - 2,
   'ytick.labelsize': fontsize - 2,
   'text.usetex': False,
   'figure.figsize': figsize,
   }
    plt.rcParams.update(params)
    # compute the medians and 25/75 percentiles
    medi, perc_25, perc_75 = perc(metric)
    # generate the x
    n_generations = metric.shape[1]
    x = numpy.arange(0, n_generations)
    
    plt.figure(figsize = figsize)
    plt.plot(x, medi, linewidth=2, color='#006BB2')
    
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.title(title, fontsize = fontsize)
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.grid()
    plt.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color='#B22400')
    plt.draw()
    if saveplot:
        plt.savefig(savingname + '.pdf', format  ='pdf', dpi = 1500)