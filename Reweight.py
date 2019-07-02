#!/usr/bin/python2.7
#########################################################################
#   Written by:                                                         #
#   Abhishek Acharya                                                    #
#   Research Associate                                                  #
#   CSIR-Central Food Technological Research Institute                  #
#   &                                                                   #
#   Visiting Student, Department of Chemistry                           #
#   IIT Kanpur                                                          #
#   Email: abhi117acharya@gmail.com                                     #
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
            print("Found {} CV fields in the colvar file.".format(len(self.legends)))
            for cv in range(len(self.legends)):
                print("CV_{}: {}".format(cv, self.legends[cv]))
        else:
            print('No legends section found in the output file. Required by the script.')
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
            print("Found {} fields in the colvar file.".format(len(self.legends)))
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
    
    In case of WS-METAD simulations, currently only 2 CVs can be used.
    
    Imp. Note: The biasfactor above is being calculated taking 
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
        self.pos=hills_dat[:,1]
        self.sig2=hills_dat[:,2]*hills_dat[:,2]
        self.ht=hills_dat[:,3]
        self.mtd_steps=hills_dat.shape[0]
        self.colvar_dat=colvar_dat[:,1:]
        self.colvar_time=np.array(colvar_dat[:,0], 'int')
        self.md_steps=colvar_dat.shape[0]
        self.T=T
        self.T0=T0
        self.dT=params["DELTA_TB"]
        self.ncvs=params["NCVS"]
        self.cv_str=params["CV_STRIDE"]
        self.periodicity=params["PERIODICITY"]
        self.numb=params["NUMB"]
        self.grid_params=params["GRID_PARAMS"]
        self.mtd_cv=params["MTD_CV"]
        self.mtd_str=params["MTD_STRIDE"]
        if params["TMAX"]==None:
            self.tmax=self.md_steps-1
            print("Note: TMAX value was not provided in the input file. Using TMAX=Total MD steps.")
        else:
            self.tmax=params["TMAX"]
        self.tmin=params["TMIN"] 
        self.mtd_grid=self.grid_params[self.mtd_cv]
        self.mtd_means=create_floats(self.mtd_grid[0], 
                                     self.mtd_grid[1], 
                                     self.mtd_grid[2])
        self.ngrids=[ int((i[1]-i[0])/i[2])+1 for i in self.grid_params]
        self.mtd_ngrid = self.ngrids[self.mtd_cv]
        self.biasfact=float((self.T0+self.dT))/self.T0 
        self.alpha=self.biasfact/(self.biasfact-1)
        self.kbt=self.T*KB
        self.gkbt = self.kbt*self.biasfact
        
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
            diff=PBC_update(cvpos-pos)
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
        self._setPBC_()
        self.ct=np.zeros(self.mtd_steps)
        self.ht=temp_unitconv(self.ht, 'kcal').reshape(self.mtd_steps,1)
        grid=np.tile(self.mtd_means, [self.mtd_steps,1])
        diff=grid-self.pos.reshape(self.mtd_steps,1)
        diff=PBC_update(diff)
        fes=-self.ht*np.exp(-(diff*diff*0.5)/self.sig2.reshape(self.mtd_steps,1))
        for i in range(self.mtd_steps):
            tmp_fes=fes[:i+1,:]
            tmp_fes=np.sum(tmp_fes, axis=0)
            temp_facs=np.sum(np.exp(-tmp_fes/self.kbt))/np.sum(np.exp(-tmp_fes/self.gkbt))
            self.ct[i]=self.kbt*np.log(temp_facs)
        
    def calc_vbias(self):
        """Calculates V(s(R), t)."""
        cv_pos=self.colvar_dat[:,self.mtd_cv]
        step_fac=self.mtd_str/self.cv_str
        rem_pos = np.array(cv_pos[-1])
        cv_pos=cv_pos[:-1].reshape(self.mtd_steps,step_fac)
        self.vbias = np.array([self._hill_estimator_(cv_pos[i], i, self.pos, self.sig2, self.ht) for i in range(self.mtd_steps)])
        self.vbias=self.vbias.flatten()
        self.vbias=np.append(self.vbias, self._hill_estimator_(rem_pos, self.mtd_steps, self.pos, self.sig2, self.ht))
    
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

class MultipleWalker(MTD_Reweighting):
    pass

class Reweight:
    """Implements reweighting for TASS, WS-METAD and US-dAFED simulation data."""
    def __init__(self, method, params, T0, outputdir):
        """Sets intitializations."""
        self.method=method
        self.params=params
        self.T0=T0
        self.outputdir=outputdir
    
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
                print("Warning: More than one CV has been detected. Yet, the --choose_cv flag has not been used. By default, the script will run calculations on the first CV. Rerun calculations for your choice of cv's by using --choose_cv")
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
                 print("Warning: More than two CV's have been detected in the colvar file. Yet, the --choose_cv flag has not been used. By default, the script will run calculations on the first two CV's. Rerun calculations for your choice of cv's by using --choose_cv")
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
            print('No specific CVs were specified. The code assumes that all the fields are corresponding to a CV. The corresponing GRID_PARAMS for all CVS are compulsary though.')
        
        if endframe==0:
            framecount = len(data)
            data = data[beginframe:framecount]
        
        print('Note: We are generating a N-dimentional histogram for the input data. The output can be dumped in a file at the final stage, but for visualization it needs further processing.')
        print('Lets print the useful data.')
        print(data)
        print(gridpars)
        print(periodicity)
        print(f.legends)
        #a method to do one shot PBC update
        
        out_indices = [find_bin(PBC_update(data[:,i]), gridpars[i][0], gridpars[i][2]) if periodicity[i]==True else find_bin(data[:,i], gridpars[i][0], gridpars[i][2]) for i in range(len(f.legends))]
        print(out_indices)
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
    
    def METAD_adiabatic(self, extT, colvar_file, hills_file):
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
        colvar=ReadColvar(colvar_file)
        hills=ReadHills(hills_file)
        metad=MTD_Reweighting(colvar.data, hills.data, self.T0, self.T, self.params)
        metad.calc_ct()
        metad.calc_vbias()
        metad.calc_prob()
        self.P_unbiased=self._normalize_histo_(metad.prob, metad.grid_params[:,2])
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

################### Functions ####################

def arg_sanity_check(args):
    """Sanity check for commandline arguments."""
    if args.method==None:
        raise Exception("The method argument was not provided. Provide the method using '-m' or '--method' flag.")
    elif args.method.upper() not in ["TASS", "WS", "US-AFED", "US", "METAMD"]:
        raise Exception("Wrong Input method provided. Check input for flag '-m' or '--method'.")
    else:
        print("{} reweighting will be performed to obtain unbiased probability distributions.".format(args.method))
    
    if args.folder: 
        args.folder=glob.glob(args.folder+"*")
        if len(args.folder) == 0:
            raise Exception("No folders were found in the run directory. Check input for flag '-f' or '--folder'")
    else:
        print("The flag --folder has a value of None. The code will assume that no umbrella sampling has been performed. Currently, reweighting for METAD runs combined with TAMD is implemented")
        print("Checking if chosed method is METAMD.")
        if args.method.upper()!="METAMD":
            raise Exception("Wrong choice of method. Either, you need to provide the name of the umbrella folders (if there are multiple US simulation runs) or the --method must be METAMD.")
        else:
            print("OK")
    
    if not os.path.exists(args.param_file):
        raise Exception("No paramfile named {} was found in the run directory. Check input for flag '-p' or '--param_file.'".format(args.param_file))
    
    if args.Temp0:
        print("Using system temperature: {}".format(args.Temp0))
    
    if not args.Temp and args.method.upper() in ["TASS", "US-AFED", "METAMD"]:
        raise Exception("Method is {} but temperature for extended system not provided. Use '-T' or '--Temp' flag.".format(args.method))
    elif args.Temp and args.method.upper() in ["US", "WS"]:
        print("Extended temperature not needed. Not using --Temp value.")
    else:
        print("Using CV temperature: {}".format(args.Temp))
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
        param_table={"NCVS": 0,
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
                if argflags[0] in ["NCVS", "MTD_CV", "CV_STRIDE", "MTD_STRIDE", "TMIN", "TMAX", "DELTA_TB"]:
                    param_table[argflags[0]]=int(argflags[1])
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
        if method=='WS' and param_table['NCVS'] > 2:
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
        param_table= {"NCVS": 0,
                     "PERIODICITY":None,
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
                if argflags[0] in ["NCVS", "MTD_CV", "CV_STRIDE", "MTD_STRIDE", "TMIN", "TMAX", "DELTA_TB"]:
                    param_table[argflags[0]]=int(argflags[1])
                elif argflags[0]=="PERIODICITY":
                    param_table[argflags[0]]=tuple([boolarg for boolarg in argflags[1].split()])
                elif argflags[0]=="GRID_PARAMS":
                    griddat = [range_flags(gridarg) for gridarg in argflags[1].split()]
                    param_table[argflags[0]]=np.array(griddat)
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
    return np.linspace(start,end,nbins,endpoint=True)

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


##################### Main ######################
def main(args):
    """
    Implements the main flow of the code.
    
    NOTE: In case of US-AFED and US, the iteration over umbrella folders are being 
    done in an inner loop. This is a little problematic if we want to choose the type
    reweighting we want for each umbrella. This feature may be implemented in a future 
    version, but will require some extensive changes to the code, so it may take a while.
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
        ofold = glob.glob('bck.*.'+outdir)
        if len(ofold) == 0:
            os.system("mv {} bck.1.{}".format(outdir, outdir))
        else:
            n = int(ofold[-1][4])
            os.system("mv {} bck.{}.{}".format(outdir, n+1, outdir))
        os.mkdir(outdir)
    
    reweight = Reweight(args.method, inparams, args.Temp0, outdir)
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
                cmd1 = "mv PhND.dat PhND_{}.dat".format(Ucv)
                print("Unbiased distribution written in {}".format("PhND_{}.dat".format(Ucv)))
                cmd2 = "cp PhND_{}.dat ../{}".format(Ucv, outdir)
            os.system(cmd1)
            os.system(cmd2)
            os.chdir("../")
    elif args.method == "US-AFED":
        reweight.US_adiabatic(args.Temp, folders, args.colvar_file, args.beginframe, args.endframe)
    elif args.method == "METAMD":
        reweight.METAD_adiabatic(args.Temp, args.colvar_file, args.hills_file)
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
            cmd1 = "mv PhND.dat PhND_{}.dat".format(Ucv)
            print("Unbiased distribution written in {}".format("PhND_{}.dat".format(Ucv)))
            cmd2 = "cp PhND_{}.dat ../{}".format(Ucv, outdir)
        os.system(cmd1)
        os.system(cmd2)
    else:
        reweight.US(folders, args.colvar_file, args.beginframe, args.endframe) #checked
    
    print("Done!!")
    return None
        

if __name__=="__main__":
    main(inarg)
