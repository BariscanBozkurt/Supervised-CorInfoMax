import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math, display
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
########### LATEX Style Display Matrix ###############
def display_matrix(array):
    """Display given numpy array with Latex format in Jupyter Notebook.
    Args:
        array (numpy array): Array to be displayed
    """
    data = ""
    for line in array:
        if len(line) == 1:
            data += " %.3f &" % line + r" \\\n"
            continue
        for element in line:
            data += " %.3f &" % element
        data += r" \\" + "\n"
    display(Math("\\begin{bmatrix} \n%s\\end{bmatrix}" % data))

def med(data):
    median = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.median(data[:, i])
    return median

def perc(data):
    # Quartiles are computed with numpy in the same way as the mean, but using the function percentile.
    mean = np.zeros(data.shape[1])
    perc_25 = np.zeros(data.shape[1])
    perc_75 = np.zeros(data.shape[1])
    std = np.zeros(data.shape[1])
    for i in range(0, len(mean)):
        mean[i] = np.mean(data[:, i])
        perc_25[i] = np.percentile(data[:, i], 25)
        perc_75[i] = np.percentile(data[:, i], 75)
        std[i] = np.std(data[:,i])
    return mean, perc_25, perc_75, std

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
    title = 'Number of Trial : {}, \nBest Result : {:.3f}, Worst Result : {:.3f}, \nMean : {:.3f}, Median : {:.3f}'.format(last_results.shape[0],np.max(last_results), np.min(last_results), np.mean(last_results), np.median(last_results))
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
    x = np.arange(0, n_generations)
    
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

def plot_regression_quality(target, pred, xlabel ='Target', ylabel = 'Predicted',figsize = (15,9), label_fontsizes = 30, tick_label_fontsizes = 30):
    """
    target : Given target values for the regression problem. Can be pandas dataframe or numpy array.
    pred   : Predictions resulting from the developed model. Can be pandas dataframe or numpy array.
        
    The function visualizes the predictions to compare the errors between target and predictions.
    """
    
    try:
        target = target.values # In case the given targets are pandas dataframe, we convert it to numpy
    except:
        pass

    try:
        pred = pred.values  # In case the given predictions are pandas dataframe, we convert it to numpy
    except:
        pass

    plt.figure(figsize=figsize)
    plt.scatter(x=target, y=pred)
    plt.plot(target, target,'r')
    rho = pd.Series(target.reshape(-1,),dtype = float).corr(pd.Series(pred.reshape(-1,), dtype = float)) # Calculate the R-squared metric

    rmse_tst = np.sqrt(mse(target, pred)) #Calculate root mean squared error
    mae_tst = mae(target, pred) # Calculate mean absolute error

    plt.title(f'RMSE: {round(rmse_tst,2)}, MAE: {mae_tst}', fontsize =  label_fontsizes) # Write the resulting metrics in the title
    plt.xlabel(xlabel, fontsize =  label_fontsizes)
    plt.ylabel(ylabel, fontsize =  label_fontsizes)
    plt.xticks(fontsize=tick_label_fontsizes)
    plt.yticks(fontsize=tick_label_fontsizes)
    plt.grid()
    plt.show()

def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)

def ApplyFont(ax, xlabel_text_size = 25.0, ylabel_text_size = 25.0, title_text_size = 19.0, x_ticks_text_size = 20, yticks_text_size = 20):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()
    text_size = 20.0
    
    for t in ticks:
        t.set_fontname('Times New Roman')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(xlabel_text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(ylabel_text_size)

    txt = ax.get_xticks()
    txt_xlabel = txt
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(x_ticks_text_size)
    
    txt = ax.get_yticks()
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(yticks_text_size)
    
    
    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(yticks_text_size)