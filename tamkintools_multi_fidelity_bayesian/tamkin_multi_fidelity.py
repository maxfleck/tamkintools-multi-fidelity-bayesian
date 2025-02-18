import numpy
numpy.testing.Tester = False
import numpy as np
np.random.seed(20)
import GPy
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel
import matplotlib.pyplot as plt

class tamkin_multi_fidelity():
    
    def __init__(self, angles, energies, weights, cmap="winter", 
                 colors=["cornflowerblue","k","magenta"], angle_range=180.0 ):
        """
        initializes the multi fidelity object
        
        angles:   np.array
            angles of a rotor scan
        energies: np.array
            corresponding energies
        weights:  np.array
            based on the weights, fidelities are estimated. Zero weights are excluded.
        cmap: string
            colormap for plots. Default is winter because its snowing outside my office.
        """
        
        self.cmap = cmap
        self.colors = colors
        
        self.msize=12 
        self.fsize=15
        self.alpha=0.3
        self.lsize = 4
        
        self.analytical_gradient_prediction = False
        
        self.fidelity_labels = ["low fidelity data","high fidelity data"]
        
        # exclude zero weights
        weights  = np.array(weights)
        p = np.where( weights > 0.)
        if len(p) < len(weights):
            print( "zero weights excluded" )
        
        # keep nonzero weights
        self.fidelities = np.array(weights)[p]
        self.angles     = np.array(angles)[p]
        self.energies   = np.array(energies)[p]
        
        # assign fidelities based on weights
        self.unique_weights = np.sort(np.unique(self.fidelities))
        self.no_fidelities = len(self.unique_weights)
        print("number of fidelities:", self.no_fidelities)
        for i,f in enumerate(self.unique_weights):
            self.fidelities[ np.where(self.fidelities==f) ] = i
        print(self.fidelities)
    
        # norm data for stable fits
        self.angle_min    = np.min(self.angles)
        if angle_range:
            self.angle_range = angle_range
            print("angle range:" , np.max(self.angles) - self.angle_min)
            print("set angle range", angle_range)
        else:
            self.angle_range  = np.max(self.angles) - self.angle_min
        self.energy_min   = np.min(self.energies)
        self.energy_range = np.max(self.energies) - self.energy_min
        self.normed_angles   = ( self.angles - self.angle_min )/ self.angle_range
        self.normed_energies = ( self.energies - self.energy_min )/ self.energy_range
        
        # plot normed data and estimated fidelites
        plt.scatter(self.normed_angles, self.normed_energies,c=self.fidelities, cmap=self.cmap)
        plt.xlabel("normalised angle")
        plt.ylabel("normalised energy")
        plt.show()
        plt.close()
        
        # assign periodicity and plot periodic normed data and estimated fidelites
        self.periodic_angles     = np.concatenate((self.normed_angles-1, 
                                                   self.normed_angles, 
                                                   self.normed_angles+1), axis=0)
        self.periodic_energies   = np.tile(self.normed_energies, 3)
        self.periodic_fidelities = np.tile(self.fidelities, 3)
        plt.scatter(self.periodic_angles, self.periodic_energies,c=self.periodic_fidelities, cmap=self.cmap)
        plt.xlabel("periodic normalised angles")
        plt.ylabel("periodic normalised energies")
        plt.show()
        plt.close()
        return
    
    def multi_fidelity(self, kernel=GPy.kern.RBF,n_optimization_restarts=5,dims=1):
        """
        initializes the multi fidelity object
        
        kernel:  GPy.kern
            kernel used in the model (one of the same type for every fidelity)
            the default kernel is the RBF kernel: https://gpy.readthedocs.io/en/deploy/GPy.kern.html
            for physics the behavior of the matern kernel might be better suited than RBF. We need to test this.
            GPy.kern.Matern32 or 52
        n_optimization_restarts: int
            number of optimization restarts with different starting inis
        dims:  int
            dimensions of the multi fidelity model
        """
        Y_train = np.atleast_2d( self.periodic_energies ).T
        X_train = np.array( (self.periodic_angles, self.periodic_fidelities) ).T
        print(X_train.shape)
        print(Y_train.shape)
        #print(X_train)
        #print(Y_train)
        
        kernels = []
        for _ in range(self.no_fidelities):
            kernels.append( kernel(dims) )
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, 
                                                       n_fidelities=self.no_fidelities)       
        self.lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, self.no_fidelities, 
                                                  n_optimization_restarts=n_optimization_restarts)
        return
        
    def train(self):
        """
        trains the multi fidelity object.
        
        Adjust and fix hyperparameters and stuff between initialisation and training of the model
        """
        self.lin_mf_model.optimize()    
        return

    def predict(self, x, fidelity="high" ):
        """
        predicts normed results for a trained multi fidelity model
        
        x:  np.array
            normed input data (values between 0 and 1)
            
        returns mean predictions and corresponding variances as np.arrays
        """     
        #xx  = np.atleast_2d(x).T
        #xxx = convert_x_list_to_array([x, x])
        #xxxx =  xxx[len(x):]        
        
        dummy = np.atleast_2d(x).T
        X_plot = convert_x_list_to_array([dummy, dummy])
        if fidelity=="low":
            X_plot = X_plot[:len(dummy)]
        else:
            X_plot = X_plot[len(dummy):]        
        
        hf_mean, hf_var = self.lin_mf_model.predict(xxxx)
        return hf_mean, hf_var
    
    def predict_hf_normed(self,dummy):
        """
        predicts normed results for a trained multi fidelity model
        
        dummy:  np.array
            normed input data (values between 0 and 1)
            
        returns mean predictions and corresponding standard deviations as np.arrays
        """
        dummy = np.atleast_2d(dummy).T
        X_plot = convert_x_list_to_array([dummy, dummy])
        X_plot_l = X_plot[:len(dummy)]
        X_plot_h = X_plot[len(dummy):]

        hf_mean, hf_var = self.lin_mf_model.predict(X_plot_h)
        hf_std = np.sqrt(hf_var)
        return hf_mean, hf_std

    def predict_hf(self,dummy):
        """
        predicts absolute results for a trained multi fidelity model
        
        dummy:  np.array
            normed input data (values between 0 and 1)
            
        returns absolute inputs, mean predictions and corresponding standard deviations as np.arrays
        """
        dummy = np.atleast_2d(dummy).T
        X_plot = convert_x_list_to_array([dummy, dummy])
        X_plot_l = X_plot[:len(dummy)]
        X_plot_h = X_plot[len(dummy):]        
        
        hf_mean, hf_std = self.predict_hf_normed(X_plot_h)
        ndummy  = dummy*self.angle_range + self.angle_min
        hf_mean = np.squeeze(hf_mean)
        hf_std  = np.squeeze(hf_std)
        ndummy  = np.squeeze(ndummy)
        return ndummy, hf_mean*self.energy_range + self.energy_min, hf_std*self.energy_range


    def plot_normed_results(self,save="", legend=True):
        """
        plots normed rsults of a trained multi fidelity model
        """
        dummy = np.linspace(0,1,100)
        hf_mean, hf_std = self.predict_hf_normed(dummy)
        hf_mean = np.squeeze(hf_mean)
        hf_std = np.squeeze(hf_std)

        for i,w in enumerate( self.unique_weights[:-1] ):
            #print(i,w)
            p = np.squeeze( np.where( self.fidelities==i ) )
            plt.plot( self.normed_angles[p], self.normed_energies[p],".", 
                     color=self.colors[i], label=self.fidelity_labels[i],
                    markersize=self.msize)        
        plt.plot(dummy, hf_mean,color=self.colors[-1], label="high fidelity pred",linewidth=self.lsize)
        plt.fill_between(dummy, hf_mean-hf_std, hf_mean+hf_std,alpha=self.alpha,color=self.colors[-1])
        #plt.scatter(self.angles, self.energies,c=self.fidelities, cmap=self.cmap)
        
        i = len(self.unique_weights)-1
        w = self.unique_weights[i]
        p = np.squeeze( np.where( self.fidelities==i ) )
        plt.plot( self.normed_angles[p], self.normed_energies[p],".", 
                 color=self.colors[i], label=self.fidelity_labels[i],
                markersize=self.msize)           
        
        plt.xlim(0,1)
        plt.xlabel("normalised angle",fontsize=self.fsize)
        plt.ylabel("normalised energy",fontsize=self.fsize)
        plt.xticks(fontsize=self.fsize)     
        plt.yticks(fontsize=self.fsize)     
        if legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,2,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                fontsize=self.fsize,frameon=False ) #,loc='center left', bbox_to_anchor=(1.01, 0.5))
        if save:
            plt.savefig(save+".png", bbox_inches='tight')
            plt.savefig(save+".pdf", bbox_inches='tight')
        plt.show()
        plt.close()        
        return        
    
    
    def plot_results(self,save="",xlabel="angle / °",ylabel="energy / (kJ/mol)", legend=True):
        """
        plots absolute rsults of a trained multi fidelity model
        """
        dummy = np.linspace(0,1,100)
        ndummy, hf_mean, hf_std = self.predict_hf(dummy)
        
        fig, ax = plt.subplots()
        ax.yaxis.offsetText.set_fontsize(self.fsize)
        ax.ticklabel_format(useOffset=False)
        for i,w in enumerate( self.unique_weights[:-1] ):
            #print(i,w)
            p = np.squeeze( np.where( self.fidelities==i ) )
            plt.plot( self.angles[p], self.energies[p],".", 
                     color=self.colors[i], label=self.fidelity_labels[i],
                    markersize=self.msize)
        plt.plot(ndummy, hf_mean,color=self.colors[-1], label="high fidelity pred",linewidth=self.lsize)
        plt.fill_between(ndummy, hf_mean-hf_std, hf_mean+hf_std,alpha=self.alpha,color=self.colors[-1])
        #plt.scatter(self.angles, self.energies,c=self.fidelities, cmap=self.cmap)

        i = len(self.unique_weights)-1
        w = self.unique_weights[i]        
        p = np.squeeze( np.where( self.fidelities==i ) )
        plt.plot( self.angles[p], self.energies[p],".", 
                 color=self.colors[i], label=self.fidelity_labels[i],
                markersize=self.msize)        
        
        dummy = np.sort( np.array((0,1))*self.angle_range + self.angle_min )
        plt.xlim(dummy[0],dummy[1])
        plt.xlabel(xlabel,fontsize=self.fsize)
        plt.ylabel(ylabel,fontsize=self.fsize)
        plt.xticks(fontsize=self.fsize)     
        plt.yticks(fontsize=self.fsize)     
        if legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,2,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                fontsize=self.fsize,frameon=False ) #,loc='center left', bbox_to_anchor=(1.01, 0.5))
        if save:
            plt.savefig(save+".png", bbox_inches='tight')
            plt.savefig(save+".pdf", bbox_inches='tight')
        plt.show()
        plt.close()        
        return       
