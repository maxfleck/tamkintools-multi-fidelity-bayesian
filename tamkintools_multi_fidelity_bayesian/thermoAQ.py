import numpy
numpy.testing.Tester = False
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.util.general import get_quantiles
import numpy as np
import matplotlib.pyplot as plt

class AcquisitionThermo(AcquisitionBase):
    """
    Thermo acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, beta=1.0, jitter=1.0, plot=False,
                 optimizer=None, cost_withGradients=None, savepath=""):
        self.optimizer = optimizer
        self.beta   = beta
        self.plot = plot
        self.savepath = savepath
        self.call_count = 0
        self.msize=8 
        self.fsize=15
        self.alpha=0.3
        self.lsize = 4                
        super(AcquisitionThermo, self).__init__(model, space,optimizer=optimizer,
                                                   cost_withGradients=cost_withGradients)
        self.jitter = jitter

    @staticmethod
    def fromConfig(model, space,plot, optimizer, cost_withGradients, config):
        return AcquisitionThermo(model, beta=config['beta'], plot=plot,
                                    optimizer=optimizer, cost_withGradients=cost_withGradients,
                                    jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes Thermo acquisition function

        """
        m, v = self.model.predict(x)
        p    = 1/np.sqrt( 2*np.pi*v )
        lnp  = p*np.log( p )
        mb   = self.beta*np.exp( -m*self.beta )
        f_acqu = mb - self.jitter*lnp

        if np.sum( x.shape ) > 4 and self.plot:
            
            plt.plot(x,self.jitter*lnp,".",label="entropic",markersize=self.msize)
            plt.plot(x,mb,".",label="energetic",markersize=self.msize)            
            plt.plot(x,f_acqu,".",label="total",markersize=self.msize)
            
            plt.legend(fontsize=self.fsize,frameon=False )
            plt.xlabel("normalised angle",fontsize=self.fsize)
            plt.ylabel("acquisition",fontsize=self.fsize)
            plt.xticks(np.linspace( np.min(x), np.max(x),6 )  , np.around(np.linspace( 0, 1, 6),1) ,fontsize=self.fsize)     
            plt.yticks( fontsize=self.fsize)         
            plt.xlim(np.min(x),np.max(x))      
            if self.savepath:
                plt.savefig(self.savepath+str(self.call_count)+".png", bbox_inches='tight')
                plt.savefig(self.savepath+str(self.call_count)+".pdf", bbox_inches='tight')                
            plt.show()
            plt.close()
            self.call_count += 1
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes Thermo acquisition function and its derivative (has a very easy derivative!)

        """
        m, v , dmdx, dvdx = self.model.predict_withGradients(x)
        p    = 1/np.sqrt( 2*np.pi*v )
        lnp  = p*np.log( p )
        mb   = self.beta*np.exp( -m*self.beta )
        f_acqu = mb - self.jitter*lnp

        dmb  = -dmdx*np.exp( -m*self.beta ) + self.beta*np.exp( -m*self.beta )
        dlnp = dvdx*( np.log( 2*np.pi*v ) - 2 )/ (4*np.sqrt(2*np.pi*v)*v )
        df_acqu = dmb + self.jitter*dlnp
        return f_acqu, df_acqu
        
        
class AcquisitionThermoDyn(AcquisitionBase):
    """
    Thermo acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, beta=1.0, m_range=1.0, jitter=1.0, plot=False,
                 optimizer=None, cost_withGradients=None, savepath=""):
        self.optimizer = optimizer
        self.beta_ini  = beta
        self.beta      = beta
        self.m_range_ini   = m_range
        self.m_range   = m_range
        self.plot = plot
        self.savepath = savepath
        self.call_count = 0
        self.msize=8 
        self.fsize=15
        self.alpha=0.3
        self.lsize = 4        
        super(AcquisitionThermoDyn, self).__init__(model, space,optimizer=optimizer,
                                                   cost_withGradients=cost_withGradients)
        self.jitter = jitter
        
    @staticmethod
    def fromConfig(model, space,plot, optimizer, cost_withGradients, config):
        return AcquisitionThermoDyn(model, beta=config['beta'], plot=plot,
                                    optimizer=optimizer, cost_withGradients=cost_withGradients,
                                    jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes Thermo acquisition function

        """
        
        m, v = self.model.predict(x)
        
        if len(x) > 10:
            # compute new beta
            m_range_dummy = abs( np.max(m) - np.min(m) )
            print(m_range_dummy , self.m_range)
            self.beta = self.beta*m_range_dummy / self.m_range
            self.m_range = m_range_dummy
            print( "new beta" , self.beta)
            # compute new Z
            self.Z = np.mean( np.exp( - self.beta * np.squeeze(m) ) )
            print( "new Z" , self.Z )        
        
        p    = 1/np.sqrt( 2*np.pi*v )
        lnp  = p*np.log( p )
        mb   = self.beta*np.exp( -m*self.beta )
        f_acqu = mb - self.jitter*lnp

        if np.sum( x.shape ) > 4 and self.plot:
           
            plt.plot(x,self.jitter*lnp,".",label="entropic",markersize=self.msize)
            plt.plot(x,mb,".",label="energetic",markersize=self.msize)            
            plt.plot(x,f_acqu,".",label="total",markersize=self.msize)
            
            plt.legend(fontsize=self.fsize,frameon=False )
            plt.xlabel("normalised angle",fontsize=self.fsize)
            plt.ylabel("acquisition",fontsize=self.fsize)
            plt.xticks(np.linspace( np.min(x), np.max(x),6 )  , np.around(np.linspace( 0, 1, 6),1) ,fontsize=self.fsize)    
            plt.yticks( fontsize=self.fsize)              
            if self.savepath:
                plt.savefig(self.savepath+str(self.call_count)+".png", bbox_inches='tight')
                plt.savefig(self.savepath+str(self.call_count)+".pdf", bbox_inches='tight')                
            plt.show()
            plt.close()
            self.call_count += 1
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes Thermo acquisition function and its derivative (has a very easy derivative!)

        """
        m, v , dmdx, dvdx = self.model.predict_withGradients(x)
        p    = 1/np.sqrt( 2*np.pi*v )
        lnp  = p*np.log( p )
        mb   = self.beta*np.exp( -m*self.beta )
        f_acqu = mb - self.jitter*lnp

        dmb  = -dmdx*np.exp( -m*self.beta ) + self.beta*np.exp( -m*self.beta )
        dlnp = dvdx*( np.log( 2*np.pi*v ) - 2 )/ (4*np.sqrt(2*np.pi*v)*v )
        df_acqu = dmb + self.jitter*dlnp
        return f_acqu, df_acqu        
