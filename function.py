import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def plot_autocorr( pe_obj ):
    pe_id_string = pe_obj.mc_names[0]
    W = pe_obj.e_windowsize[pe_id_string]
    W_n = np.squeeze( range(0,2*W) ) 
    rho = np.squeeze([pe_obj.e_rho[pe_id_string][:2*W]])
    d_rho = np.squeeze([pe_obj.e_drho[pe_id_string][:2*W]])
    tau_int = np.squeeze([pe_obj.e_n_tauint[pe_id_string][:2*W]])
    dtau_int = np.squeeze([pe_obj.e_n_dtauint[pe_id_string][:2*W]])
    matplotlib.rcParams['figure.figsize'] = [15, 5]
    fig,axs = plt.subplots( 1,2 )
    axs[0].axhline(0, linestyle="dashed",color="gray")
    axs[0].axvline(W, linestyle="dashed",color="orange")
    axs[0].plot(W_n, rho, label=pe_id_string)
    axs[0].set_xlabel(r"$\Delta\tau$")
    axs[0].legend(loc="upper right")
    axs[1].axvline(W, linestyle="dashed",color="orange")
    axs[1].errorbar(W_n, tau_int, yerr=dtau_int, label=pe_id_string)
    axs[1].axhline([pe_obj.e_tauint[pe_id_string]], linestyle="dashed",color="gray")
    axs[1].legend(loc="lower right")
    
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y
    
def lin_fit( x, y, dy, y_win_regulator = 0.1):
    a_fit , cov = curve_fit( linearFunc, x , y, sigma = dy, absolute_sigma=True)
    inter = a_fit[0]
    slope = a_fit[1]
    d_inter = np.sqrt(cov[0][0])
    d_slope = np.sqrt(cov[1][1])
    yfit = inter + slope*np.linspace(0., np.max(x)*1.2, num=100, endpoint=True)
    fig, axs = plt.subplots(figsize=(10, 5))
    axs.plot( np.linspace(0., np.max(x)*1.2, num=100, endpoint=True) , yfit, label='Fit')
    axs.errorbar( x , y, yerr = dy, fmt='r.', label='Data')
    axs.errorbar( 0 , inter, yerr=d_inter, marker='d', label='estrapped')
    #axs.set_ylim(y[3]*(1-y_win_regulator),y[3]*(1+y_win_regulator))
    axs.legend()
    return [inter,slope,d_inter,d_slope]

def blocking( measurement, block_size, title="blocking for your data"):
    
    temporary_variance = np.zeros( np.size(block_size) )
    
    for i in range( np.size(block_size) ):
        current_block_size = block_size[i]
        current_number_of_data = np.size(measurement) - np.size(measurement)% current_block_size
        current_data = measurement[0:current_number_of_data]
        current_data = np.reshape( current_data, (-1,current_block_size)  )
        current_data = np.mean(current_data,1)
        temporary_variance[i] = np.var(current_data)/np.size(current_data)
        
    temporary_variance[0] = np.var(measurement)/np.size(measurement)
    center_of_array = int( (np.size(block_size) - np.size(block_size)%2)/2) 
    true_variance = np.mean(temporary_variance[center_of_array:])
    plt.plot(block_size, temporary_variance,'-b')
    plt.plot(block_size, temporary_variance,'.r')
    plt.plot(block_size, np.ones(np.size(block_size))*true_variance, "--k", label="estimated var")
    plt.xlabel("number of elements per block")
    plt.ylabel(r"$\sigma^2$")
    plt.legend(loc="lower right")
    plt.title(title)
    plt.show()
        
        
    return np.sqrt(true_variance)