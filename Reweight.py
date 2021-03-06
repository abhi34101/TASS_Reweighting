#!/usr/bin/python2.7
#########################################################################
#   Written by:                                                         #
#   Abhishek Acharya                                                    #
#   Graduate Student                                                    #
#   Jacobs University Bremen                                            #
#   Email: a.acharya@jacobs-university.de                               #
#                                                                       #
#   {The TASS and WS parts of the code written while working as:        #
#   Research Associate                                                  #
#   CSIR-Central Food Technological Research Institute                  #
#   &                                                                   #
#   Visiting Student, Department of Chemistry                           #
#   IIT Kanpur}                                                         #
#                                                                       #
#########################################################################
###################### Import libraries ##########################

try:
        
    import numpy as np
except:
    raise Exception("The script requires numpy. Install the latest numpy library.")
import glob, os
import argparse

################# Parse commandline arguments ####################
parser = argparse.ArgumentParser(description="A script to reweight TASS (N-D), WS(2-D), US-dAFED [2-D] data.")
parser.add_argument("-m","--method", action='store', default=None, type=str, help="Simulation method employed. 'WS', 'TASS', 'METAMD', 'US-AFED' or 'US'.")
parser.add_argument("-f", "--folder", action="store", default=None, type=str, help="String common to folders containing the COLVAR files. Default: None")
parser.add_argument("-cvf", "--colvar_file", action="store", default="COLVAR", type=str, help="Name of the colvar file in each folder. Default: COLVAR")
parser.add_argument("-hlf", "--hills_file", action="store", default="HILLS", type=str, help="Name of the hills file in each folder, Default: HILLS")
parser.add_argument("-p", "--param_file", action="store", default="reweight.par", type=str, help="Name of the input parameter file.")
parser.add_argument("-T", "--Temp", action="store", type=int, default=None, help="Temperature of the extended system in Kelvin.")
parser.add_argument("-T0", "--Temp0", action="store", default=300, type=int, help="Temperature of the real system in Kelvin. Default: 300")
parser.add_argument("-b", "--beginframe", action="store", default=1, type=int, help=" starting frame number to use in calculations. Data-points corresponding to frame before -b will not be used for calculations.")
parser.add_argument("-e", "--endframe", action="store", default=0, type=int, help="Final frame number for use in calculations. By default read from the colvar file.")
parser.add_argument("-cnv", "--chkconv", action="store", type=bool, default=False, help="Set this flag to True for checking convergence.")
parser.add_argument("-bs", "--blksize", action="store", type=int, default=0, help="number of data points included successively for checking convergence.")
parser.add_argument("-mtz", "--mintozero", action="store", type=bool, default=False, help="Set this flag to True for setting global minima or chosen minima to zero.")
parser.add_argument("-mtzc", "--mtzcoord", action="store", default=None, help="The pmf grid coordinate for the chosen minima to reset to zero. By default, the code uses the global minima. Usage (for a 2D pmf): --mtzcoord 'x, y'")
parser.add_argument("-mtzs", "--mtzgsize", action="store", default=None, help="The square grid size to use for choosing the region around the minima for resetting to zero. Usage: --mtzgsize '5'")

inarg=parser.parse_args()
################ Globals ######################## 

KB=1.9872041E-3

################ Classes ########################

class ReadColvar:
    """Reads plumed COLVAR file."""
    def __init__(self, filename):
        self.filename=filename
        self.data = np.loadtxt(self.filename)
        self.legends=None
        
    def read_field(self):
        with open(self.filename) as f:
            content = f.readlines()
        for l in list(content):
            if "FIELDS" in l and l.startswith("#"):
                self.legends = l.split()[3:]
            else:
                break
        if self.legends != None:
            print(">>> Found {} fields in the colvar file.".format(len(self.legends)))
            for cv in range(len(self.legends)):
                print("CV_{}: {}".format(cv, self.legends[cv]))
        else:
            print('>>> No legends section found in the output file. Required by the script.')
        return None

class ReadHills(ReadColvar):
    """Reads plumed HILLS file."""
        
    def read_field(self):
        with open(self.filename) as f:
            content = f.readlines()
        for l in list(content):
            if "FIELDS" in l and l.startswith("#"):
                self.legends = l.split()[2:]
            else:
                break
        if self.legends!=None:
            print(">>> Found {} fields in the hills file.".format(len(self.legends)))
        return None

class MTD_Reweighting:
    """
    Implements Tiwary and Parrinello method [JPCB (2015) 119, 736-742] for 
    reweighting metadynamics runs perfomed within the framework of 
    WS-METAD [Awasthi S., Kapil V. and Nair N.N. JCC (2016), 37, 1413]  
    and TASS [Awasthi S. and Nair N.N. JCP (2017) 146, 094108].
    
    This works for TASS simulations with any number of collective variables 
    but is currently limited to reweighting simulations with metadynamics 
    bias along one orthogonal metad CV across all umbrellas. The other CVs are 
    sampled by coupling them to the extended system at higher temperature 
    for which a separate reweighting step is used. Note: The Umbrella and METAD 
    CV are also put under Temperature Acceleration which makes reweighting easier.
    
    In case of WS-METAD simulations, only 2 CVs can be used (1 US 
    and 1 METAD).
    
    Imp. Note: The biasfactor is being calculated taking 
    Physical temp - self.T0  (and not self.T as it should be since the 
    METAD is done in the extended high temperature space) as this implicitly 
    accounts for dAFED reweighting step in TASS. Also note that the inverse
    temp (beta) value is calculated using the extended temp (self.T).
    In case of WS-MTD, the code resets the extended temp (self.T) to 
    T0 to obtain correct values for inverse temp (beta). See code in
    class Reweight.
    """
    def __init__(self, colvar_dat, hills_dat, T0, T, params):
        """Setting some initializations"""
        print("Metadynamics Reweighting step.")
        self.pos=hills_dat[:,1]                           #position of the hills
        self.sig2=hills_dat[:,2]*hills_dat[:,2]           #sigma value
        self.ht=hills_dat[:,3]                            #hill height
        self.mtd_steps=hills_dat.shape[0]                 #no. of metad steps
        self.cvs=params["CVS"]                            #choice of cvs
        try:
            self.ncvs=len(params["CVS"])                  #number of CVs
        except:
            raise Exception('>>> No CVS are provided in the input paramfile. The code will exit now.')
        self.colvar_dat=colvar_dat[:,1:][:,tuple(self.cvs)]    #select only the required CVS
        self.colvar_time=np.array(colvar_dat[:,0], 'int') #reads in the time column
        self.md_steps=colvar_dat.shape[0]                 #total md steps (not metad)
        self.T=T                                          #the extended system temperature (TAMD bias)
        self.T0=T0                                        #the real system temperature
        self.dT=params["DELTA_TB"]                        #delta T value (WT-METAD bias)
        self.cv_str=params["CV_STRIDE"]                   #print stride used to print colvar files
        self.periodicity=params["PERIODICITY"]            #periodicity of the chosen CVs
        self.grid_params=params["GRID_PARAMS"]            #params for the grid construction
        self.mtd_cv=params["MTD_CV"]                      #CV index for the METAD CV. This needs to be the index value of MTD CV from within the chosen CVS
        self.mtd_str=params["MTD_STRIDE"]                 #PACE used for METAD sampling
        if params["TMAX"]==None:                          #here we choose the range of data to select for calculation, ensuring that data falls in the quasistationary regime
            self.tmax=self.md_steps-1
            print(">>> Note: TMAX value was not provided in the input file. Using TMAX=Total MD steps.")
        else:
            self.tmax=params["TMAX"]
        self.tmin=params["TMIN"]                           #this is more relevant to the previous comment
        self.mtd_grid=self.grid_params[self.mtd_cv]        #here we create the metad CV grid using the grid params data
        self.mtd_means=create_floats(self.mtd_grid[0],     #the mean position for each grid on the metad grid is calculated
                                     self.mtd_grid[1],
                                     self.mtd_grid[2])
        self.ngrids=[ int((i[1]-i[0])/i[2])+1 for i in self.grid_params] #grid length along all the chosen CVs dimensions
        self.mtd_ngrid = self.ngrids[self.mtd_cv]          #grid length along the METAD CV dimension
        self.biasfact=float((self.T0+self.dT))/self.T0     #biasfactor as calculated base on the delta T value in the param file
        self.alpha=self.biasfact/(self.biasfact-1)         #this is a factor required for vbias calculation (related ot biasfactor)
        self.kbt=self.T*KB                                 #the temperature factor (beta_inv)
        self.gkbt = self.kbt*self.biasfact                 #beta_inv multiplied by the biasfactor
        
    def _setPBC_(self):
        """
        Updates the Metad-CV positions(from HILLS file) and the MD-CV postions (from COLVAR file).
        """
        if self.periodicity[self.mtd_cv]==True:
            self.pos = PBC_update(self.pos)
        for i in range(self.ncvs):
            if self.periodicity[i]: 
                self.colvar_dat[:,i]=PBC_update(self.colvar_dat[:,i])
        return None
        
    def _hill_estimator_(self, cvpos, ntimes, pos, sig2, ht):
        """Calculates the V(s(R), t) contributions."""
        if ntimes==0:
            return np.zeros_like(cvpos)
        else:
            cvpos=np.tile(cvpos, [ntimes,1])
            pos=pos[:ntimes].reshape(ntimes,1)
            sig2=sig2[:ntimes].reshape(ntimes,1)
            ht=ht[:ntimes].reshape(ntimes,1)/self.alpha
            if self.periodicity[self.mtd_cv]==True:
                diff=PBC_update(cvpos-pos)                   #PBC update
            hillgrid = ht*np.exp(-(diff*diff*0.5)/sig2)   
            return np.sum(hillgrid, axis=0)
    
    def _find_bin_(self, arr, gridparams):
        """
        Given an n-dimensional input array of values and the gridparams, provides the indices of the values on the grid.
        Note: not adding +1 to indices as counting starts from zero in python.
        """
        ncvs = arr.shape[1]
        gridmins=np.array(gridparams)[:,0].reshape(1,ncvs)
        widths=np.array(gridparams)[:,2].reshape(1,ncvs)
        indices=(arr-gridmins)/widths
        return indices.astype(int)

    def calc_ct(self):
        """c(t) calculation step"""
        self._setPBC_()                         #checks the periodicity info and updates CV values and METAD hill pos values accordingly
        self.ct=np.zeros(self.mtd_steps)        #initializes an array for storing c(t) values with the size equal to the number of metad steps.
        self.ht=temp_unitconv(self.ht, 'kcal').reshape(self.mtd_steps,1)  #unit conversion of hill height to kcal from kJ  and necessary reshaping
        grid=np.tile(self.mtd_means, [self.mtd_steps,1])  #creates a grid of dimension mtd_grid_dimension*metad_steps
        diff=grid-self.pos.reshape(self.mtd_steps,1)      #calculate (sgridmean-smtd)
        if self.periodicity[self.mtd_cv]==True:
            diff=PBC_update(diff)                         #PBC update
        fes=-self.ht*np.exp(-(diff*diff*0.5)/self.sig2.reshape(self.mtd_steps, 1)) #calculate bias contributions to each grid for each mtd step.
        for i in range(self.mtd_steps):                   #loop over all MTD columns to calculate the c(t) value after each mtd step
            tmp_fes=fes[:i+1,:]
            tmp_fes=np.sum(tmp_fes, axis=0)
            temp_facs=np.sum(np.exp(-tmp_fes/self.kbt))/np.sum(np.exp(-tmp_fes/self.gkbt))
            self.ct[i]=self.kbt*np.log(temp_facs)
        
    def calc_vbias(self):
        """Calculates V(s(R), t)."""
        cv_pos=self.colvar_dat[:,self.mtd_cv]              #values of mtd CV at each MD step taken from COLVAR file
        step_fac=self.mtd_str/self.cv_str                  #the step factor taking into account the mtd pace and colvar print stride
        init_pos = np.array(cv_pos[:step_fac])                     #the initial cv position. This is printed after the final hill addition.
        cv_pos=cv_pos[step_fac:].reshape(self.mtd_steps,step_fac) #reshaping the cv_pos array to make a mtdsteps*stepfac array
        self.vbias = np.array(np.zeros_like(init_pos)) #looping over each MTD column to calculate the bias contributions of the MTD hill at each COLVAR position
        hiltmp=[self._hill_estimator_(cv_pos[i-1], i, self.pos, self.sig2, self.ht) for i in range(1,self.mtd_steps)]
        mtd_vbias=np.array(hiltmp)
        self.vbias=np.append(self.vbias, mtd_vbias)
    
    def calc_prob(self):
        """Constructs the final metad unbiased probability (un-normalized)."""
        self.prob = np.zeros(self.ngrids)
        colvar=self.colvar_dat[self.tmin:self.tmax,:]
        indices = self._find_bin_(colvar, self.grid_params)
        vbias=self.vbias[self.tmin:self.tmax]
        time = self.colvar_time[self.tmin:self.tmax]
        for i in range(len(vbias)):
            num = np.exp((vbias[i]-self.ct[time[i]])/self.kbt)
            np.add.at(self.prob, tuple(indices[i]), num)
        
class MultipleWalker:
    """Implements the Metadynamics reweighting for the mutliple walker case. Reads bias from colvar file as dumped by plumed."""
    def __init__(self, colvar_dat, T0, T, params):
        """Setting some initializations"""
        print("Metadynamics Reweighting step.")
        self.cvs=params["CVS"]                            #choice of cvs
        self.nwalker=params["NWALKER"]
        self.dT=params["DELTA_TB"]                        #delta T value (WT-METAD bias)
        self.cv_str=params["CV_STRIDE"]                   #print stride used to print colvar files
        self.periodicity=params["PERIODICITY"]            #periodicity of the chosen CVs
        self.grid_params=params["GRID_PARAMS"]            #params for the grid construction
        self.mtd_cv=params["MTD_CV"]                      #CV index for the METAD CV. This needs to be the index value of MTD CV from within the chosen CVS
        self.mtd_str=params["MTD_STRIDE"]                 #PACE used for METAD sampling
        self.colvar_dat=colvar_dat[:,tuple(self.cvs)]     #select only the required CVS
        self.colvar_time=np.array(colvar_dat[:,0], 'int') #reads in the time column
        self.md_steps=colvar_dat.shape[0]                 #total md steps (not metad)
        if params["TMAX"]==None:                          #here we choose the range of data to select for calculation, ensuring that data falls in the quasistationary regime
            self.tmax=self.md_steps
            print(">>> Note: TMAX value was not provided in the input file. Using TMAX=Total MD steps.")
        else:self.dT=params["DELTA_TB"]                        #delta T value (WT-METAD bias)
        self.cv_str=params["CV_STRIDE"]                   #print stride used to print colvar files
        self.periodicity=params["PERIODICITY"]            #periodicity of the chosen CVs
        self.grid_params=params["GRID_PARAMS"]            #params for the grid construction
        self.tmin=params["TMIN"]
        self.rbiascol=params["RBIASCOL"]
        self.mtd_steps=self.md_steps/self.nwalker
        try:
            self.ncvs=len(params["CVS"])                  #number of CVs
        except:
            raise Exception('No CVS are provided in the input paramfile. The code will exit now.')
        
        self.T=T                                          #the extended system temperature (TAMD bias)
        self.T0=T0                                        #the real system temperature
                                   #this is more relevant to the previous comment
        
        self.ngrids=[ int((i[1]-i[0])/i[2])+1 for i in self.grid_params] #grid length along all the chosen CVs dimensions#grid length along the METAD CV dimension
        self.biasfact=float((self.T0+self.dT))/self.T0     #biasfactor as calculated base on the delta T value in the param file
        self.alpha=self.biasfact/(self.biasfact-1)         #this is a factor required for vbias calculation (related ot biasfactor)
        self.kbt=self.T*KB                                 #the temperature factor (beta_inv)
        self.gkbt = self.kbt*self.biasfact                 #beta_inv multiplied by the biasfactor
        if params["WALLS"]:
            self.walls=params["WALLS"]
            self.wallbias=colvar_dat[:,tuple(self.walls)]
            self.wallbias=temp_unitconv(np.sum(self.wallbias, axis=1), 'kcal')
        self.rbias_plumed=temp_unitconv(colvar_dat[:,self.rbiascol], 'kcal')
        

    def _find_bin_(self, arr, gridparams):
        """
        Given an n-dimensional input array of values and the gridparams, provides the indices of the values on the grid.
        Note: not adding +1 to indices as counting starts from zero in python.
        """
        ncvs = arr.shape[1]
        gridmins=np.array(gridparams)[:,0].reshape(1,ncvs)
        widths=np.array(gridparams)[:,2].reshape(1,ncvs)
        indices=(arr-gridmins)/widths
        return indices.astype(int)
        
# =============================================================================
#     def _hill_estimator_(self, cvpos, ntimes, pos, sig2, ht):
#         """Calculates the V(s(R), t) contributions."""
#         cvpos=np.tile(cvpos, [ntimes,1,1])
#         pos=pos[:ntimes].reshape(ntimes,1,self.nwalker)
#         sig2=sig2[:ntimes].reshape(ntimes,1,self.nwalker)
#         ht=ht[:ntimes].reshape(ntimes,1,self.nwalker)/self.alpha
#         diff=cvpos-pos
#         if self.periodicity[self.mtd_cv]==True:
#             diff=PBC_update(diff)                   #PBC update
#         hillgrid = ht*np.exp(-(diff*diff*0.5)/sig2)   
#         return np.sum(hillgrid, axis=0)
#         self.mtd_ngrid = self.ngrids[self.mtd_cv]  
#     def calc_ct(self):
#         """calculate c(t)"""
#         self._setPBC_()
#         self.ct=np.zeros(self.mtd_steps)
#         self.ht=temp_unitconv(self.ht, 'kcal').reshape(self.mtd_steps,self.nwalker, 1)
#         grid=np.tile(self.mtd_means, [self.mtd_steps, self.nwalker, 1])
#         self.pos=self.pos.reshape(self.mtd_steps, self.nwalker, 1)
#         self.sig2=self.sig2.reshape(self.mtd_steps, self.nwalker, 1)
#         diff=grid-self.pos
#         if self.periodicity[self.mtd_cv]==True:
#             diff=PBC_update(diff)
#         fes=-self.ht*np.exp(-(diff*diff*0.5)/self.sig2)
#         fes=np.sum(fes, axis=1)
#         for i in range(self.mtd_steps):                   #loop over all MTD columns to calculate the c(t) value after each mtd step
#             tmp_fes=fes[:i+1]                              ###We can also sum over the walkers at the first step, outside this for loop maybe that would be correct.
#             tmp_fes=np.sum(tmp_fes, axis=0)
#             temp_facs=np.sum(np.exp(-tmp_fes/self.kbt))/np.sum(np.exp(-tmp_fes/self.gkbt))
#             self.ct[i]=self.kbt*np.log(temp_facs)
#         self.ct = np.append(np.array([0.0]), self.ct)
#         
#     
#     def calc_vbias(self):
#         """Calculates V(s(R), t)."""
#         cv_pos=self.colvar_dat[:,self.mtd_cv]             #values of mtd CV at each MD step taken from COLVAR file
#         step_fac=self.mtd_str/self.cv_str                  #the step factor taking into account the mtd pace and colvar print stride
#         init_pos = np.array(cv_pos[:(self.md_steps%(self.mtd_steps*step_fac*self.nwalker))])                     #the final cv position. This is printed after the final hill addition.
#         cv_pos=cv_pos[(self.md_steps%(self.mtd_steps*step_fac*self.nwalker)):].reshape(self.mtd_steps,step_fac,self.nwalker) #reshaping the cv_pos array to make a mtdsteps*stepfac array
#         self.vbias = np.array([np.zeros_like(init_pos)])
#         #looping over each MTD column to calculate the bias contributions of the MTD hill at each COLVAR position
#         hiltmp=[self._hill_estimator_(cv_pos[i-1], i, self.pos, self.sig2, self.ht) for i in range(1,self.mtd_steps+1)]
#         mtd_vbias=np.array(hiltmp)
#         mtd_vbias=mtd_vbias.flatten()
#         self.vbias=self.vbias.flatten()
#         self.vbias=np.append(self.vbias, mtd_vbias)
# =============================================================================
        
    def calc_prob(self):
        """Constructs the final metad unbiased probability (un-normalized)."""
        self.prob = np.zeros(self.ngrids)
        colvar=self.colvar_dat[self.tmin:self.tmax,:]
        self.wallbias=self.wallbias[self.tmin:self.tmax]
        self.rbias_plumed=self.rbias_plumed[self.tmin:self.tmax]
        indices = self._find_bin_(colvar, self.grid_params)
        for i in range(len(self.rbias_plumed)):
            num=np.exp((self.rbias_plumed[i]+self.wallbias[i])/self.kbt)
            np.add.at(self.prob, tuple(indices[i]), num)

class Reweight:
    """Implements reweighting for TASS, WS-METAD, METAMD (METAD+TAMD), US and 
       US- simulation data.
    """
    def __init__(self, method, params, T0, outputdir,  conv=False, blksize=0):
        """Sets intitializations."""
        self.method=method
        self.params=params
        self.T0=T0
        self.outputdir=outputdir
        self.pmf=None
        self.conv=conv
        self.blksize=blksize
        if self.conv==True and self.method=="METAMD":
            self.params["TMIN"]=0
            print("Convergence check turned on. Resetting TMIN to zero")
    
    def _normalize_histo_(self, p_hist, widths):
        """Normalizes the probability histogram given the grid bin-width in each dimesion."""
        for i in widths:
            p_hist/=i
        return p_hist/p_hist.sum()
    
    def _histo1d_(self, gridpar1, colvar_file, beginframe, endframe, choose_cv=None, periodicity=(False)):
        """Constructs a 1D histogram for the given CV."""
        minv1, maxv1, bin_width1=gridpar1
        cv1_keys = create_floats(minv1, maxv1, bin_width1)
        histo =  np.zeros(len(cv1_keys))
        f = ReadColvar(colvar_file)
        f.read_field()
        ncvs=f.legends
        data = f.data[:,1:]
        if endframe==0:
            framecount = len(data)
            data = data[beginframe:framecount]
        
        if choose_cv:
            if ncvs>1:
                cvindex=choose_cv
            else:
                cvindex=[0]
        else:
            if ncvs>1:
                print(">>> Warning: More than one CV has been detected. Yet, the --choose_cv flag has not been used. By default, the script will run calculations on the first CV. Rerun calculations for your choice of cv's by using --choose_cv")
            cvindex=[0]
        
        print("Creating 1D-histogram of CV1...")
        c1=data[:,cvindex[0]]
        if periodicity[0]==True:
            c1 = PBC_update(c1)
        c1_indices=find_bin(c1, minv1, bin_width1)
        np.add.at(histo, (c1_indices), 1)
        return histo, cv1_keys
            
    def _histo2d_(self, gridpar1, gridpar2, colvar_file, beginframe, endframe, choose_cv=None, periodicity=(False,False)):
        """Constructs a 2D histogram for the given CVs."""
        minv1, maxv1, bin_width1=gridpar1
        minv2, maxv2, bin_width2=gridpar2
        cv1_keys = create_floats(minv1, maxv1, bin_width1) 
        cv2_keys = create_floats(minv2, maxv2, bin_width2)
        histo =  np.zeros([len(cv1_keys),len(cv2_keys)])
        f = ReadColvar(colvar_file)
        f.read_field()
        ncvs=f.legends
        data = f.data[:,1:]
        if endframe==0:
            framecount = len(data)
            data = data[beginframe:framecount]
            
        if choose_cv:
            if ncvs>2:
                cvindex=choose_cv
            else:
                cvindex=[0, 1]
        else:
            if ncvs>2:
                 print(">>> Warning: More than two CV's have been detected in the colvar file. Yet, the --choose_cv flag has not been used. By default, the script will run calculations on the first two CV's. Rerun calculations for your choice of cv's by using --choose_cv")
            cvindex=[0, 1]
        
        print("Creating 2D-histogram of CV{}-vs-CV{}...".format(cvindex[0], cvindex[1]))
        c1=data[:,cvindex[0]]
        c2=data[:,cvindex[1]]
        if periodicity[0]==True:
            c1 = PBC_update(c1)
        if periodicity[1]==True:
            c2 = PBC_update(c2)
        c1_indices=find_bin(c1, minv1, bin_width1)
        c2_indices=find_bin(c2, minv2, bin_width2)
        np.add.at(histo, (c1_indices, c2_indices), 1)
        return histo, cv1_keys, cv2_keys
    
    def _histoNd_(self, gridpars, colvar_file, beginframe, endframe, choose_cv=None, periodicity=None):
        """Constructs an N-dimensional histogram. This can accept high 
        dimensional data, but functionality may be limited by the available 
        memory.
        """
        cv_keys = [create_floats(i[0], i[1], i[2]) for i in gridpars]
        lengths = [len(x) for x in cv_keys]
        histo = np.zeros(lengths)
        f = ReadColvar(colvar_file)
        f.read_field()
        data=f.data[:,1:]
        
        if choose_cv:
            print("Selecting only the CVS specified in input file..")
            data = data[:,tuple(choose_cv)]
            f.legends=[list(f.legends)[i] for i in choose_cv]
        else:
            print('>>> No specific CVs were specified. The code assumes that all the fields are corresponding to a CV. The corresponing GRID_PARAMS for all CVS are compulsary though.')
        
        if endframe==0:
            framecount = len(data)
            data = data[beginframe:framecount]
        
        print('>>> Note: We are generating a N-dimentional histogram for the input data. The output can be dumped in a file at the final stage, but for visualization it needs further processing.')
        out_indices = [find_bin(PBC_update(data[:,i]), gridpars[i][0], gridpars[i][2]) if periodicity[i]==True else find_bin(data[:,i], gridpars[i][0], gridpars[i][2]) for i in range(len(f.legends))]
        np.add.at(histo, tuple(out_indices), 1)
        return histo, cv_keys
    
    def US(self, folders, colvar_file, beginframe, endframe):
        """Implements simple histogramming of US data."""
        status=self.params["PERIODICITY"]
        cvs=self.params["CVS"]
        if len(cvs)==2:
            gridpar1=self.params["GRID_PARAMS"][0]
            gridpar2=self.params["GRID_PARAMS"][1]
            for fold in folders:
                os.chdir(fold)
                Hh, cv1keys, cv2keys = self._histo2d_(gridpar1, gridpar2, colvar_file, beginframe, endframe, choose_cv=cvs, periodicity=status)
                self.Ph_unbiased=self._normalize_histo_(Hh, [gridpar1[2], gridpar2[2]])
                write2Dhisto(self.Ph_unbiased, cv1keys, cv2keys)
                if '_' in fold:
                    Ucv = fold.split("_")[1]
                else:
                    Ucv = fold
                cmd3 = "mv Ph2D.dat Ph2D_{}.dat".format(Ucv)
                os.system(cmd3)
                cmd4 = "cp Ph2D_{}.dat ../{}".format(Ucv, self.outputdir) 
                os.system(cmd4)
                os.chdir("../")
        else:
            gridpar1=self.params["GRID_PARAMS"][0]
            for fold in folders:
                os.chdir(fold)
                Hh, cv1keys = self._histo1d_(gridpar1, colvar_file, beginframe, endframe, choose_cv=cvs, periodicity=status)
                self.Ph_unbiased=self._normalize_histo_(Hh, [gridpar1[2]])
                write1Dhisto(self.Ph_unbiased, cv1keys)
                if '_' in fold:
                    Ucv = fold.split("_")[1]
                else:
                    Ucv = fold
                cmd3 = "mv Ph1D.dat Ph1D_{}.dat".format(Ucv)
                os.system(cmd3)
                cmd4 = "cp Ph1D_{}.dat ../{}".format(Ucv, self.outputdir) 
                os.system(cmd4)
                os.chdir("../")
        return None
        
    def US_adiabatic(self, extT, folders, colvar_file, beginframe, endframe): #Now should work for N-dimensions; needs testing though.
        ##Issue: higher the number of dimensions, greater is the amount of memory required. The maximum dimensions that can be used therefore
        ##depends on the available system memory.
        """Implements simple histogramming of US data with dAFED reweighting."""
        self.T = extT
        status=self.params["PERIODICITY"]
        cvs=self.params["CVS"]
        gridparams = self.params['GRID_PARAMS']
        for fold in folders:
            os.chdir(fold)
            Hh, cvkeys = self._histoNd_(gridparams, colvar_file, beginframe, endframe, choose_cv=cvs, periodicity=status)
            Ph=dAFED_reweighting(Hh, self.T0, self.T)
            self.Ph_unbiased=self._normalize_histo_(Ph, [i[2] for i in gridparams])
            if '_' in fold:
                Ucv = fold.split("_")[1]
            else:
                Ucv = fold
            if len(cvkeys)==1:
                write1Dhisto(self.Ph_unbiased, cvkeys[0])
                cmd3 = "mv Ph1D.dat Ph1D_{}.dat".format(Ucv)
                os.system(cmd3)
                print("Unbiased distribution written in {}".format("Ph1D_{}.dat".format(Ucv)))
                cmd4 = "cp Ph1D_{}.dat ../{}".format(Ucv, self.outputdir)
                os.system(cmd4)
            elif len(cvkeys)>2:
                writeNDhisto(self.Ph_unbiased)
                cmd3 = "mv PhND.dat PhND_{}.dat".format(Ucv)
                os.system(cmd3)
                print("Unbiased distribution written in {}".format("PhND_{}.dat".format(Ucv)))
                cmd4 = "cp PhND_{}.dat ../{}".format(Ucv, self.outputdir)
                os.system(cmd4)
            else:
                write2Dhisto(self.Ph_unbiased, cvkeys[0], cvkeys[1])
                cmd3 = "mv Ph2D.dat Ph2D_{}.dat".format(Ucv)
                os.system(cmd3)
                print("Unbiased distribution written in {}".format("Ph2D_{}.dat".format(Ucv)))
                cmd4 = "cp Ph2D_{}.dat ../{}".format(Ucv, self.outputdir)
                os.system(cmd4)
            os.chdir("../")
        print("Done!!")
        return None
    
    def METAD_adiabatic(self, extT, colvar_file, outdir, mintozero, mtzcrd, mtzgsize):
        """
        Implements reweighting for simulations biased using WT-METAD and TAMD.
        This assumes that METAD is performed along only one CV and TAMD is used
        to bias sampling along all the other orthogonal CVs. Although in 
        principle there is no limit to the number of CV that can be biased, but 
        there may be problems in running the reweighting step with large number
        of CVs. Practically, it makes sense to just do reweighting for the main
        CVs of interest, that can be visualized (puts a limit of max 3 CVs).
        """
        self.T=extT
        kbt0=KB*self.T0
        colvar=ReadColvar(colvar_file)
        if self.conv==True:
            #loop run
            endpt=int((colvar.data.shape[0]/self.params["NWALKER"])/10)*self.params["NWALKER"]
            while endpt < colvar.data.shape[0]:
                nucolvar=colvar.data[:endpt,:]
                metad=MultipleWalker(nucolvar, self.T0, self.T, self.params)
                metad.calc_prob()
                self.P_unbiased=self._normalize_histo_(dAFED_reweighting(metad.prob, metad.T0, metad.T), metad.grid_params[:,2])
                self.pmf=-kbt0*np.log(self.P_unbiased)
                
                if mintozero:
                    self.pmf=min_to_zero(self.pmf, mtzcrd, mtzgsize)
                    
                endptout=(nucolvar[-1,0]/1000000)*self.params["NWALKER"]
                if len(self.params["GRID_PARAMS"])==1:
                    minv, maxv, bin_width=self.params["GRID_PARAMS"][0]
                    cv_keys=create_floats(minv, maxv, bin_width)
                    write1Dhisto(self.P_unbiased, cv_keys)
                    print(">>> Unbiased distribution written to Ph1D.dat.")
                    cmd1="mv Ph1D.dat {:}/Ph1D_{:0.2f}.dat".format(outdir, endptout)
                    writeFE(self.pmf, np.array([cv_keys]), "FESout_1D.dat")
                    cmd2="mv FESout_1D.dat {:}/FESout_1D_{:0.2f}.dat".format(outdir, endptout)
                    os.system(cmd1)
                    os.system(cmd2)
                elif len(self.params["GRID_PARAMS"])==2:
                    minv1, maxv1, bin_width1=self.params["GRID_PARAMS"][0]
                    minv2, maxv2, bin_width2=self.params["GRID_PARAMS"][1]
                    cv1_keys = create_floats(minv1, maxv1, bin_width1)
                    cv2_keys = create_floats(minv2, maxv2, bin_width2)
                    write2Dhisto(self.P_unbiased, cv1_keys, cv2_keys)
                    print(">>> Unbiased distribution written in Ph2D.dat")
                    cmd1 = "mv Ph2D.dat {:}/Ph2D_{:0.2f}.dat".format(outdir, endptout)
                    writeFE(self.pmf, np.array([cv1_keys, cv2_keys]), 'FESout_2D.dat')
                    cmd2="mv FESout_2D.dat {:}/FESout_2D_{:0.2f}.dat".format(outdir, endptout)
                    os.system(cmd1)
                    os.system(cmd2)
                else:
                    raise Exception("Convergence check not implemented for CV dimensions greater than 2.")
                endpt+=self.blksize*self.params["NWALKER"]
        else:
            metad=MultipleWalker(colvar.data, self.T0, self.T, self.params)
            metad.calc_prob()
            self.P_unbiased=self._normalize_histo_(dAFED_reweighting(metad.prob, metad.T0, metad.T), metad.grid_params[:,2])
            self.pmf=-kbt0*np.log(self.P_unbiased)
            
            if mintozero:
                self.pmf=min_to_zero(self.pmf, mtzcrd, mtzgsize)
                    
            if len(self.params["GRID_PARAMS"])==2:
                minv1, maxv1, bin_width1=self.params["GRID_PARAMS"][0]
                minv2, maxv2, bin_width2=self.params["GRID_PARAMS"][1]
                cv1_keys = create_floats(minv1, maxv1, bin_width1)
                cv2_keys = create_floats(minv2, maxv2, bin_width2)
                write2Dhisto(self.P_unbiased, cv1_keys, cv2_keys)
                print(">>> Unbiased distribution written in Ph2D.dat")
                cmd1 = "mv Ph2D.dat {}".format(outdir)
                writeFE(self.pmf, np.array([cv1_keys, cv2_keys]), 'FESout_2D.dat')
                cmd2="mv FESout_2D.dat {}".format(outdir)
                os.system(cmd1)
                os.system(cmd2)
            elif len(self.params["GRID_PARAMS"])==3:
                minv1, maxv1, bin_width1=self.params["GRID_PARAMS"][0]
                minv2, maxv2, bin_width2=self.params["GRID_PARAMS"][1]
                minv3, maxv3, bin_width3=self.params["GRID_PARAMS"][2]
                cv1_keys = create_floats(minv1, maxv1, bin_width1)
                cv2_keys = create_floats(minv2, maxv2, bin_width2)
                cv3_keys = create_floats(minv3, maxv3, bin_width3)
                write3Dhisto(self.P_unbiased, cv1_keys, cv2_keys, cv3_keys)
                print(">>> Unbiased distribution written in Ph3D.dat")
                cmd1 = "mv Ph3D.dat {}".format(outdir)
                writeFE(self.pmf, np.array([cv1_keys, cv2_keys, cv3_keys]), 'FESout_3D.dat')
                cmd2="mv FESout_3D.dat {}".format(outdir)
                os.system(cmd1)
                os.system(cmd2)
            else:
                writeNDhisto(self.P_unbiased)
                print(">>> Unbiased distribution written in PhND.dat")
                cmd1 = "mv PhND.dat {}".format(outdir)
                print('NOTE: N-dimensional data. No pmf will be written.')
                os.system(cmd1)
        return None
    
    def WS(self, T0, colvar_file, hills_file):
        """
        Implements reweighting of WS-METAD data. This assumes only one orthogonal 
        metad-CV used for biasing simulations across all umbrella slices. Implemen
        -ted only for two CVs.
        
        Note:self.T is reset to systems physical temperature to obtain correct 
        bias-factor values in Metadynamics reweighting step (without dAFED reweighting).
        """
        self.T=T0 
        colvar=ReadColvar(colvar_file)
        hills=ReadHills(hills_file)
        metad = MTD_Reweighting(colvar.data, hills.data, self.T0, self.T, self.params)
        metad.calc_ct()
        metad.calc_vbias()
        metad.calc_prob()
        self.P_unbiased=self._normalize_histo_(metad.prob, metad.grid_params[:,2])
        return None
    
    def TASS(self, extT, colvar_file, hills_file):
        """
        Implements reweighting of TASS data. Can be used with any number of 
        CVs, but the assumes that only one orthogonal metad-CV is used for 
        biasing simulations across all umbrella slices. Rest of the CV's are
        assumed to be sampled by high temperature of the extended system. Here
        too there is no limit to the number of CV that can be biased, but 
        there may be problems in running the reweighting step with large number
        of CVs. Practically, it makes sense to just do reweighting for the main
        CVs of interest, that need to be visualized (puts a limit of max 3 CVs).
        """
        self.T=extT
        colvar=ReadColvar(colvar_file)
        hills=ReadHills(hills_file)
        metad = MTD_Reweighting(colvar.data, hills.data, self.T0, self.T, self.params)
        metad.calc_ct()
        metad.calc_vbias()
        metad.calc_prob()
        self.P_unbiased=self._normalize_histo_(metad.prob, metad.grid_params[:,2])
        return None


################## Functions ####################
        
def min_to_zero(pmf, mtzcrd, mtzgsize):
    if len(pmf.shape)<=3:
        if not mtzcrd:
            minval=pmf.min()
            minindex=np.where(pmf==minval)
            if len(pmf.shape)==3:
                mtzcrd=list([minindex[0][0], minindex[1][0], minindex[2][0]])
                gridoffset=int(mtzgsize[0]/2)
                mingrid=pmf[mtzcrd[0]-gridoffset:mtzcrd[0]+gridoffset+1,mtzcrd[1]-gridoffset:mtzcrd[1]+gridoffset+1,mtzcrd[2]-gridoffset:mtzcrd[2]+gridoffset+1]
            elif len(pmf.shape)==2:
                mtzcrd=list([minindex[0][0], minindex[1][0]])
                gridoffset=int(mtzgsize[0]/2)
                mingrid=pmf[mtzcrd[0]-gridoffset:mtzcrd[0]+gridoffset+1,mtzcrd[1]-gridoffset:mtzcrd[1]+gridoffset+1]
            else:
                mtzcrd=list([minindex[0][0]])
                gridoffset=int(mtzgsize[0]/2)
                mingrid=pmf[mtzcrd[0]-gridoffset:mtzcrd[0]+gridoffset+1]
        else:
            if len(pmf.shape)==3:
                gridoffset=int(mtzgsize[0]/2)
                mingrid=pmf[mtzcrd[0]-gridoffset:mtzcrd[0]+gridoffset+1,mtzcrd[1]-gridoffset:mtzcrd[1]+gridoffset+1,mtzcrd[2]-gridoffset:mtzcrd[2]+gridoffset+1]
            elif len(pmf.shape)==2:
                gridoffset=int(mtzgsize[0]/2)
                mingrid=pmf[mtzcrd[0]-gridoffset:mtzcrd[0]+gridoffset+1,mtzcrd[1]-gridoffset:mtzcrd[1]+gridoffset+1]
            else:
                gridoffset=int(mtzgsize[0]/2)
                mingrid=pmf[mtzcrd[0]-gridoffset:mtzcrd[0]+gridoffset+1]
        avgmin=np.average(mingrid)
        print("The calculate average used to shift PMF:", avgmin)
        pmf=pmf-avgmin
    else:
        print("##### mintozero not implemented for pmf with dimensions higher than 3. PMF will not be updated. #####")
    return pmf
        
def arg_sanity_check(args):
    """Sanity check for commandline arguments."""
    if args.method==None:
        raise Exception("The method argument was not provided. Provide the method using '-m' or '--method' flag.")
    elif args.method.upper() not in ["TASS", "WS", "US-AFED", "US", "METAMD"]:
        raise Exception("Wrong Input method provided. Check input for flag '-m' or '--method'.")
    else:
        print(">>> {} reweighting will be performed to obtain unbiased probability distributions.".format(args.method))
    
    if args.folder: 
        args.folder=glob.glob(args.folder+"*")
        if len(args.folder) == 0:
            raise Exception("No folders were found in the run directory. Check input for flag '-f' or '--folder'")
    else:
        print(">>> The flag --folder has a value of None. The code will assume that no umbrella sampling has been performed. Currently, reweighting for METAD runs combined with TAMD is implemented.")
        print(">>> Checking if chosed method is METAMD.")
        if args.method.upper()!="METAMD":
            raise Exception("Wrong choice of method. Either, you need to provide the name of the umbrella folders (if there are multiple US simulation runs) or the --method must be METAMD.")
    
    if not os.path.exists(args.param_file):
        raise Exception("No paramfile named {} was found in the run directory. Check input for flag '-p' or '--param_file.'".format(args.param_file))
    
    if args.Temp0:
        print(">>> Using system temperature: {}".format(args.Temp0))
    
    if not args.Temp and args.method.upper() in ["TASS", "US-AFED", "METAMD"]:
        raise Exception("Method is {} but temperature for extended system not provided. Use '-T' or '--Temp' flag.".format(args.method))
    elif args.Temp and args.method.upper() in ["US", "WS"]:
        print(">>> Extended temperature not needed. Not using --Temp value.")
    else:
        print(">>> Using CV temperature: {}".format(args.Temp))
        pass
        
    if args.hills_file and args.method in ["US-AFED", "US"]:
        print("Note: HILLS file not required for US-AFED or US reweighting.")
    
    if args.beginframe or args.endframe:
        if args.method in ["TASS", "WS", "METAMD"]:
            print("Warning: The flags '--beginframe' and '--endframe' are used only in US-AFED and US reweighting.")
            print("For TASS, WS and METAMD, the choice of starting frames are controlled by the TMIN and TMAX directives in the paramfile.")
            args.beginframe, args.endframe = None, None
        else:
            print("Using data from frames {} to {}.".format(args.beginframe, args.endframe))
    
    if args.chkconv and args.method in ["US", "US-AFED", "TASS", "WS"]:
        print(">>> Note: Convergence checking is currently implemented only for method=METAMD. Turning off convergence check.")
        args.chkconv=False
    
    if args.mintozero and args.method=="METAMD":
        print(">>> Flag --mintozero set to True.")
        if not args.mtzgsize:
            print(">>> Flag --mtzgsize not used. Falling back to default gridsize of [5, 5].")
            args.mtzgsize=[5]
        else:
            args.mtzgsize=[int(args.mtzgsize)]
        if not args.mtzcoord:
            print(">>> Flag --mtzcoord not used. We will use the global minima instead.")
        else:
            coordstr=args.mtzcoord.split(",")
            args.mtzcoord=[int(i) for i in coordstr]
        print("Following values to be used for mintozero function: --mintozero {} --mtzcoord {} --mtzgsize {}.".format(args.mintozero, args.mtzcoord, args.mtzgsize))
    elif args.mintozero and args.method!="METAMD":
        print(">>> Note: --mintozero only implemented for method METAMD. Turning of mintozero flag.")
        args.mintozero=False
    else:
        print("No mintozero flag used. Recommended for comparing different FES.")

def process_INPUT(method, paramfile):
    """
    Reads in the input file and converts to a hash-table for quick lookup.
    """
    
    def range_flags(flags):
        flags=flags[1:-1].split(",")
        return np.array(flags, 'float')
    
    def cv_process(field):
        if field.startswith("{"):
            out = field[1:-1].split(",")
            out = [int(f) for f in list(out)]
        elif field[0].isdigit():
            out = int(field)
        else:
            raise Exception("Problem with input format.")
        return out
        
    def TASS_WS_param(paramfile):
        """Parses the TASS/WS input file."""
        param_table={"CVS": None,
                     "PERIODICITY":None,
                     "NUMB": 0,
                     "GRID_PARAMS": None,
                     "MTD_CV": 1,
                     "CV_STRIDE":0,
                     "MTD_STRIDE":0,
                     "TMIN":0,
                     "TMAX":None,
                     "DELTA_TB":None
                     }
        with open(paramfile, 'r') as inpf:
            params=inpf.readlines()
        
        for line in params:
            if line.startswith("#") or line.startswith("\n"):
                pass
            else:
                argflags = line.split("=")
                if argflags[0] in ["MTD_CV", "CV_STRIDE", "MTD_STRIDE", "TMIN", "TMAX", "DELTA_TB"]:
                    param_table[argflags[0]]=int(argflags[1])
                elif argflags[0]=="CVS":
                    param_table[argflags[0]]=[int(cv) for cv in argflags[1].split()]
                elif argflags[0]=="NUMB":
                    param_table[argflags[0]]=int(argflags[1])
                elif argflags[0]=="PERIODICITY":
                    param_table[argflags[0]]=tuple([boolarg for boolarg in argflags[1].split()])
                elif argflags[0]=="GRID_PARAMS":
                    griddat = [range_flags(gridarg) for gridarg in argflags[1].split()]
                    param_table[argflags[0]]=np.array(griddat)
                else:
                    print("Wrong flagname: {}. Allowed lags should be {}".format(argflags[0], list(param_table.keys())))
                    raise ValueError("Check input file.")
        if method=='WS' and len(param_table['CVS']) > 2:
            raise Exception
        return param_table
            
    def US_AFED_param(paramfile):
        """Parses the US-dAFED input file."""
        param_table = {"PERIODICITY":None,
                     "NUMB": 0,
                     "GRID_PARAMS": None,
                     "CVS":None
                     }
        with open(paramfile, 'r') as inpf:
            params=inpf.readlines()
        
        for line in params:
            if line.startswith("#") or line.startswith("\n"):
                pass
            else:
                argflags = line.split("=")
                if argflags[0]=="NUMB":
                    param_table[argflags[0]]=int(argflags[1])
                elif argflags[0]=="PERIODICITY":
                    param_table[argflags[0]]=tuple([boolarg for boolarg in argflags[1].split()])
                elif argflags[0]=="GRID_PARAMS":
                    griddat = [range_flags(gridarg) for gridarg in argflags[1].split()]
                    param_table[argflags[0]]=griddat
                elif argflags[0]=="CVS":
                    param_table[argflags[0]]=[int(cv) for cv in argflags[1].split()]
                else:
                    print("Wrong flagname: {}. Allowed flags should be {}".format(argflags[0], list(param_table.keys())))
                    raise ValueError("Check input file.")
        return param_table
    
    def METAMD_param(paramfile):
        """Parses the METAMD input param file"""
        param_table= {"CVS": None,
                     "PERIODICITY":None,
                     "GRID_PARAMS": None,
                     "MTD_CV": 0,
                     "CV_STRIDE":0,
                     "MTD_STRIDE":0,
                     "TMIN":0,
                     "TMAX":None,
                     "DELTA_TB":None,
                     "NWALKER":1,
                     "WALLS":None,
                     "RBIASCOL":None
                     }
        with open(paramfile, 'r') as inpf:
            params=inpf.readlines()
        
        for line in params:
            if line.startswith("#") or line.startswith("\n"):
                pass
            else:
                argflags = line.split("=")
                if argflags[0] in ["MTD_CV", "CV_STRIDE", "MTD_STRIDE", "TMIN", "TMAX", "DELTA_TB", "NWALKER"]:
                    param_table[argflags[0]]=int(argflags[1])
                elif argflags[0]=="CVS":
                    param_table[argflags[0]]=[int(cv) for cv in argflags[1].split()]
                elif argflags[0]=="PERIODICITY":
                    param_table[argflags[0]]=tuple([boolarg for boolarg in argflags[1].split()])
                elif argflags[0]=="GRID_PARAMS":
                    griddat = [range_flags(gridarg) for gridarg in argflags[1].split()]
                    param_table[argflags[0]]=np.array(griddat)
                elif argflags[0]=="WALLS":
                    param_table[argflags[0]]=[int(cv) for cv in argflags[1].split()]
                elif argflags[0]=="RBIASCOL":
                    param_table[argflags[0]]=int(argflags[1])
                else:
                    print("Wrong flagname: {}. Allowed lags should be {}".format(argflags[0], list(param_table.keys())))
                    raise ValueError("Check input file.")
        return param_table
    
    if method in ["TASS", "WS"]:
        params = TASS_WS_param(paramfile)
    elif method in ["US-AFED", "US"]:
        params = US_AFED_param(paramfile)
    else:
        params = METAMD_param(paramfile)
    return params

def write1Dhisto(matrix, gridlist1):
    """Writes 1D histogram matrix to a file."""
    with open("Ph1D.dat", 'w') as out:
        for i in range(matrix.shape[0]):
            l = (" "*8).join([format(gridlist1[i], '.16f'), format(matrix[i], '.16f')])
            out.write(l+'\n')
    return None

def write2Dhisto(arr, gridlist1, gridlist2):
    """Writes 2D histogram matrix to a file."""
    with open("Ph2D.dat", 'w') as out:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                l = (" "*8).join([format(gridlist1[i], '.16f'), format(gridlist2[j], '.16f'), format(arr[i, j], '.16f')])
                out.write(l+'\n')
            out.write("\n")
    return None

def write3Dhisto(arr, gridlist1, gridlist2, gridlist3):
    """Writes 2D histogram matrix to a file."""
    with open("Ph3D.dat", 'w') as out:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    l = (" "*8).join([format(gridlist1[i], '.16f'), format(gridlist2[j], '.16f'), format(gridlist3[k]), format(arr[i, j, k], '.16f')])
                    out.write(l+'\n')
                out.write('\n')
    return None

def writeNDhisto(arr):
    """Saves numpy n-dimensional arrays as numpy binary file. The reason is because 
       arrays greater than 2-dimesions are inherently difficult to read and write\
       in a human readable manner (also slower). For extracting the data for 
       visualization purposes, a simple script for data extraction and visualization 
       will soon be provided. In addition this also saves the data in a plain one value/line format for 
       compatibility with Shalini's WHAM script'. This will soon be deprecated."""
    np.save("PhND.npy", arr)
    print("Writing numpy binary output file.")
    out=arr.flatten()
    out=list(out.astype(str))
    with open("PhND.dat", 'w') as f:
        f.write("\n".join(out))
    return None

def temp_unitconv(value, unit):
    """
    Scales the input value to the given temperature unit
    """
    if unit.lower()=="kcal":
        return 0.239006*value
    elif unit.lower()=="kj":
        return 4.184*value
    else:
        print("Warning: conversion unit not correctly provided! Returning unprocessed value.")
        return value

def PBC_update(arr):
    """Updates the value of the CV by applying PBC"""
    def update_lower(a):
        return np.where(a<-3.1416, a+6.2832, a)
    
    def update_upper(a):
        return np.where(a>3.1416, a-6.2832, a)
    
    return update_upper(update_lower(arr))

def create_floats(start, end, step):
    """Creates a list of float numbers with the given step"""
    nbins = (end-start)/step + 1
    return np.linspace(start,end,int(nbins),endpoint=True)

def find_bin(value, gridmin, step):
    """
    Finds the bin index value. Works only on a list.
    
    Note: Here, +1 is not added to ind value since in python indexing starts from 0
    """
    ind = (value-gridmin)/step
    print(ind)
    return np.array(ind, 'int') 

def dAFED_reweighting(p_hist, T0, T):
    """Performs reweighting of Probability histogram obtained from dAFED simulations."""
    tempfactor= float(T)/T0
    return p_hist**tempfactor

def writeFE(arr, grids, outf):
        with open(outf, 'w') as out:
            if len(grids)==3:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        for k in range(arr.shape[2]):
                            l = (" "*8).join([format(grids[0][i], '.16f'), format(grids[1][j], '.16f'), format(grids[2][k], '.16f'), format(arr[i, j, k], '.16f')])
                            out.write(l+'\n')
                        out.write('\n')
            elif len(grids)==2:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        l = (" "*8).join([format(grids[0][i], '.16f'), format(grids[1][j], '.16f'), format(arr[i, j], '.16f')])
                        out.write(l+"\n")
                    out.write(l+"\n")
            else:
                for i in range(arr.shape[0]):
                    l = (" "*8).join([format(grids[0][i], '.16f'), format(arr[i], '.16f')])
                    out.write(l+"\n")
                out.write("\n")
        return None


##################### Main ######################
def main(args):
    """
    Implements the main flow of the code.
    
    NOTE: In case of US-AFED and US, the iteration over umbrella folders are being 
    done in an inner loop. This is a little problematic if we want to choose the type
    reweighting we want for each umbrella. This feature may be implemented in a future 
    version, but will require some extensive changes to the code. Not a top priority now.
    
    Features:
        1. Reweighting for simulations biased using TASS and WS. c(t) and total bias is 
           calulated post simulation from the hills file, and reweighting is done using 
           the Tiwary and Parrinello method [JPCB (2015) 119, 736-742].
        2. Reweighting for simulations biased using METAD and TAMD (METAMD). In this case, 
           2D and 3D pmf can be generated for the chosen combinations of biased CVs.
        3. The METAMD reweighting reads the rbias printed by plumed into the COLVAR file.
           Make sure the rbias is printed in the COLVAR file only. This method is faster
           than recalculating rbias from scratch, as we do in case of TASS and WS.
        3. Convergence check implemented for METAMD to check convergence along one or two
           CVs.
        4. The FES generated using METAMD can be shifted to zero which allows comparing
           different FES. The code uses either the global minima or user specified grid
           corresponding to a low-energy region in the FES.
        5. The code can also generate probability distributions for US and US-TAMD 
           simulations. WHAM can be run on these distributions to get the full FES.
        6. Running this code requires two-three files. a) params file (various params for reweighting)
                                                       b) colvar file (for all methods)
                                                       c) hills file (for TASS, WS)
        
        This code is a work in progress and new features will be introduces as and when possible.
        For queries and reporting bugs, write to: a.acharya@jacobs-university.de 
    """
    try:
        arg_sanity_check(args)
    except:
        raise Exception("Something not right with input flags. Check input.")
    
    try:
        inparams=process_INPUT(args.method, args.param_file)
    except:
        raise Exception("Error in reading input parameter file.")
    
    folders=args.folder
    if folders!=None:
        print("Folders :", folders)
        if len(folders)!=inparams["NUMB"]:
            raise Exception("Number of Umbrella run directories do not match the 'NUMB' in the paramfile. Make sure you are not missing any data.")
    
    
    outdir = "Ph_outfiles_"+args.method
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)
    else:
        ofold = sorted(glob.glob('bck.*.'+outdir))
        if len(ofold) == 0:
            os.system("mv {} bck.1.{}".format(outdir, outdir))
        else:
            n = int(ofold[-1][4])
            os.system("mv {} bck.{}.{}".format(outdir, n+1, outdir))
        os.mkdir(outdir)
    
    reweight = Reweight(args.method, inparams, args.Temp0, outdir,  conv=args.chkconv, blksize=args.blksize)
    if args.method in ["TASS", "WS"]:
        for fold in folders:
            os.chdir(fold)
            if '_' in fold:
                Ucv = fold.split("_")[1]
            else:
                Ucv = fold
            if args.method=="TASS":
                reweight.TASS(args.Temp, args.colvar_file, args.hills_file)
            else:
                reweight.WS(reweight.T0, args.colvar_file, args.hills_file) #
            if len(reweight.params["GRID_PARAMS"])==2:
                minv1, maxv1, bin_width1=reweight.params["GRID_PARAMS"][0]
                minv2, maxv2, bin_width2=reweight.params["GRID_PARAMS"][1]
                cv1_keys = create_floats(minv1, maxv1, bin_width1)
                cv2_keys = create_floats(minv2, maxv2, bin_width2)
                write2Dhisto(reweight.P_unbiased, cv1_keys, cv2_keys)
                cmd1 = "mv Ph2D.dat Ph2D_{}.dat".format(Ucv)
                print("Unbiased distribution written in {}".format("Ph2D_{}.dat".format(Ucv)))
                cmd2 = "cp Ph2D_{}.dat ../{}".format(Ucv, outdir)
            else:
                writeNDhisto(reweight.P_unbiased)
                cmd1 = "mv PhNDfor a 2D pmf.dat PhND_{}.dat".format(Ucv)
                print("Unbiased distribution written in {}".format("PhND_{}.dat".format(Ucv)))
                cmd2 = "cp PhND_{}.dat ../{}".format(Ucv, outdir)
            os.system(cmd1)
            os.system(cmd2)
            os.chdir("../")
    elif args.method == "US-AFED":
        reweight.US_adiabatic(args.Temp, folders, args.colvar_file, args.beginframe, args.endframe)
    elif args.method == "METAMD":
        reweight.METAD_adiabatic(args.Temp, args.colvar_file, outdir, args.mintozero, args.mtzcoord, args.mtzgsize)
    else:
        reweight.US(folders, args.colvar_file, args.beginframe, args.endframe)
    print("Done!!")
    return None
        

if __name__=="__main__":
    main(inarg)
