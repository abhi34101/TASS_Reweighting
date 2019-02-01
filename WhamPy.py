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
import argparse
import glob
import numpy as np
import distutils.util
################# Parse commandline arguments ####################
parser =  argparse.ArgumentParser(description="Script to run WHAM on the ND probability distributions.")

parser.add_argument("-P", "--probabilityfile", action="store", default="Ph_", type=str, help="String common to files containing the probability distributions. Default: Ph_")
parser.add_argument("-I", "--wham_input", action="store", default="INPUT", type=str, help="Name of the WHAM input file. Default: INPUT. THe file contains a few flags that can be set to particular values. Some flexibility is affored while providing CV_VALUE, CV_FORCE and NFRAMES.")

inarg = parser.parse_args()
################### Functions ####################
def temp_unitconv(value, unit):
    """
    scales the input value to the given temperature unit
    
    Parameters
    ------------
    value: float or integer
        temperature value
    unit: string 'kcal' or 'kj'
        the target unit
    
    Returns
    ------------
    float after the scaling
    """
    if unit.lower()=="kcal":
        return 0.239006*value
    elif unit.lower()=="kj":
        return 4.184*value
    else:
        print "Warning: conversion unit not correctly provided! Returning unprocessed value."
        return value

def PBC_update(value):
    """
    updates the value of the CV points by applying PBC. Applicable to torsional CVs only.
    
    Parameters
    ------------
    value: float or numpy.ndarray
    any float value or a numpy n-dimensional array
    
    Returns
    ------------
    float after applying PBC
    """
    def update_lower(a):
        return np.where(a<-3.1416, a+6.2832, a)
    
    def update_upper(a):
        return np.where(a>3.1416, a-6.2832, a)
    
    if type(value)==float:
        if value > 3.1416:
            value+=-6.2832
        elif value < -3.1416:
            value+=6.2832
        else:
            pass
    elif type(value)==np.ndarray:
        value = update_upper(update_lower(value))
    else:
        raise ValueError("PBC update failed. Wrong input type.")
    return value  

def process_INPUT(wham_input):
    """
    reads in the input file and converts to a hash-table for quick lookup.
    
    Parameters:
    -------------
    wham_input: file
    This is a file in a specified format used to pass required parameters.
    
    Returns
    -------------
    dictionary or hash-table
    """
    
    def range_flags(flags):
        flags=flags[1:-1].split(",")
        nbins = round(((float(flags[1])-float(flags[0]))/float(flags[2]))+1)
        return np.linspace(float(flags[0]), float(flags[1]), nbins, endpoint=True)
    
    def cv_process(field):
        if field.startswith("{"):
            out = field[1:-2].split(",")
            out = np.array([float(f) for f in list(out)])
        elif field[0].isdigit():
            out = float(field)
        elif field.startswith("["):
            out = range_flags(field[:-1])
        else:
            raise Exception("Problem with input format.")
        return out
        
    #some initializations
    Kb=1.9872041E-3
    param_table={"NCVS": 0,
                 "PERIODICITY":None,
                 "NUMB": 0, 
                 "GRID_POINTS": None, 
                 "TEMP": 0,
                 "U_MEANS": None,
                 "U_FORCES": None,
                 "NFRAMES": None,
                 "WHAM_TOL":0}
    #read the file
    with open(wham_input, 'r') as inpf:
        params=inpf.readlines()
    for line in params:
        if line.startswith("#") or line.startswith("\n"):
            pass
        else:
            argflags = line.split("=")
            if argflags[0] in ["NCVS", "NUMB"]:
                param_table[argflags[0]]=int(argflags[1])
            elif argflags[0]=="TEMP":
                param_table[argflags[0]]=Kb*int(argflags[1])
            elif argflags[0]=="PERIODICITY":
                param_table[argflags[0]]=[bool(distutils.util.strtobool(boolarg)) for boolarg in argflags[1].split()]
            elif argflags[0]=="GRID_POINTS":
                griddat = [range_flags(gridarg) for gridarg in argflags[1].split()]
                param_table[argflags[0]]=griddat
            elif argflags[0] in ["U_FORCES", "NFRAMES", "U_MEANS"]:
                param_table[argflags[0]]=cv_process(argflags[1])
            elif argflags[0]=="WHAM_TOL":
                param_table[argflags[0]]=float(argflags[1])
            else:
                print "Wrong Flag used: {}".format(argflags[0])
                raise ValueError("Check input file.")
    #some final processing
    param_table["U_FORCES"] = np.array([temp_unitconv(param_table["U_FORCES"], 'kcal')]*param_table['NUMB'])
    if type(param_table["NFRAMES"])==np.ndarray:
        param_table["NFRAMES"] = param_table["NFRAMES"]
    elif type(param_table["NFRAMES"])==float:
        param_table["NFRAMES"] = np.array([param_table["NFRAMES"]]*param_table['NUMB'])
    else:
        raise ValueError("Error in values provided as input for flag NFRAMES. Check input file.")
    return param_table

def readProbabilityData(probabilityfile, grid):
    """
    Reads the probability data files into an n-dimensional numpy matrix.
    Flexible with any number of CVs.
    
    Note: Practically, one can visualize only 3 CVs at a time. Therefore, it
    it is recommended that at a time, a maxmimum of 3 CVs are included in the 
    probablity data file.
    
    
    Parameters
    --------------
    probabilityfile: string
        Probability file string common to all files
    grid: list
        Takes a list containing list of grid values. Required for calculating matrix dimensions.
    
    Returns
    --------------
    n-dimensional numpy matrix
    """
    def sorter(elem):
        out = elem.split("_")
        return float(out[1][:-4])
    
    files=glob.glob(probabilityfile+"*")
    files=sorted(list(files), key=sorter)
    numb=len(files)
    ncvs=len(grid)
    matrix_dim=[len(files)]
    for i in range(ncvs):
        matrix_dim.append(len(grid[i]))
    initmat=np.zeros(matrix_dim)
    
    for f in range(numb):
        print "Loading data file: .\{}".format(files[f])
        Ph_1darr = tuple(np.loadtxt(files[f], unpack=True))[-1]
        initmat[f] = Ph_1darr.reshape(matrix_dim[1:])
    return initmat
################ Classes ########################
class WHAM:
    """
    A class for running WHAM scf iterations on input probablity distributions obtained
    from independent umbrella sampling runs. It can work with upto 3 CVs.
    
    
    Parameters
    --------------
    input parameter as read from the INPUT file (See process_INPUT)
    
    Returns
    --------------
    PMF or FE surface along the CVs.
    """
    
    def __init__(self, prob, nframes, uforces, umeans, kbt, ncvs, numb, grid, periodicity, iterations, tolerance, fesout_file):
        self.data=prob
        self.kbt=kbt
        self.ncvs=ncvs
        self.numb=numb
        self.grid=grid
        self.periodicity=periodicity
        self.pmf=None
        self.ushape=[1]*(self.ncvs+1)
        self.ushape[0]=self.numb
        self.g0shape=[1]*(self.ncvs+1)
        self.g0shape[1]=self.grid[0].shape[0]
        self.nframes=nframes.reshape(self.ushape)
        self.uforces=uforces.reshape(self.ushape)
        self.umeans=umeans.reshape(self.ushape)
        self.grid0=self.grid[0].reshape(self.g0shape)
        self.feconst=np.ones(self.ushape)
        self.iterations=iterations
        self.tolerance=tolerance
        self.cnvg=1.0
        self.fesoutf=fesout_file
    
    def wham_scf(self):
        
        def remove_zeros(a):
            return np.where(a==0.0, a+10**-160, a)
        #calculate the unbiased probability using the 1st WHAM equation.
        self.gvol=1
        for g in self.grid:
            diff = np.abs(g[1]-g[0])
            self.gvol *= diff
        
        num=self.data*self.nframes  #33x65x65 matrix
        del_s0 = self.grid0-self.umeans #33x65x1 matrix
        if self.periodicity[0]==True:
            del_s0=PBC_update(del_s0)
        biash=np.exp(-(0.5*(self.uforces)*np.square(del_s0))/self.kbt) #33x65x1 matrix
        den=self.nframes*biash*self.feconst  #33x65x1 matrix
        self.Pu=np.divide(np.sum(num, axis=0), np.sum(den, axis=0)) #1x65x65 matrix
        #self.Pu=remove_zeros(self.Pu)
        
        #calculate new self.feconst_dum using 2nd WHAM equation
        self.feconst_dum=np.sum(self.gvol*self.Pu*biash, axis=(1, 2)) #1x65x65
        #calculate convergence
        self.feconst_dum=np.divide(1.0, self.feconst_dum)
        self.feconst_dum = self.feconst_dum.reshape(self.ushape)
        self.cnvg=np.sum(np.abs(np.log(self.feconst_dum)-np.log(self.feconst)))*self.kbt
        #update self.feconst to self.feconst_dum
        self.feconst = self.feconst_dum
        return None
    
    def pmfcalc(self, writepmf=False, outfile="free_energy.dat"):
        #code for calculation for pmf here
        def writeFE(arr, grid, outf):
            with open(outf, 'w') as out:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if len(grid)==3:
                            for k in range(arr.shape[2]):
                                l = (" "*8).join([format(grid[0][i], '.16f'), format(grid[1][j], '.16f'), format(grid[2][k], '.16f'), format(arr[i, j], '.16f')])
                                out.write(l+'\n')
                            out.write("\n")
                        else:
                            l = (" "*8).join([format(grid[0][i], '.16f'), format(grid[1][j], '.16f'), format(arr[i, j], '.16f')])
                            out.write(l+"\n")
                    out.write("\n")
            return None
        
        self.pmf = -self.kbt*np.log(self.Pu)
        if writepmf:
            writeFE(self.pmf, self.grid, outfile)
        return None
    
    def start(self):
        i=0
        while i <= self.iterations:
            if self.cnvg > self.tolerance:
                self.wham_scf()
                i+=1
                print "Iteration: {}   Convergence: {}".format(i, self.cnvg)
            else:
                print "Convergence Achieved !!"
                break
        else:
            print "Free energy not converged in {} iteraction.".format(self.iterations)
        print "The calculated free-energy will be written to {}.".format(self.fesoutf)
        self.pmfcalc(writepmf=True, outfile=self.fesoutf)
        return None

##################### Main ######################  
def main(args):
    """
    main code
    
    Parameters
    ------------
    args: a python class with arguments as attributes 
        commandline arguments parsed using argparse
    
    Returns
    ------------
    None
    """
    WHAM_ITERATIONS=20000
    print "Starting WhamPy..."
    print "Reading input param file..."
    try:
        inparam=process_INPUT(args.wham_input)
    except:
        raise Exception("Error in reading input parameter file.")
    
    print "Reading Probability data files..."
    indata = readProbabilityData(args.probabilityfile, inparam['GRID_POINTS'])
    
    #Implementing PBC if umbrella CV is periodic
    if inparam['PERIODICITY'][0]==True:
        inparam['U_MEANS']=PBC_update(inparam['U_MEANS'])
    
    print "Initializing WHAM calculations on probablity distribution in {}-dimensional CV space.".format(inparam['NCVS'])
    print "Number of Umbrellas to process: {}".format(inparam['NUMB'])
    print "Using WHAM scf tolerance: {}".format(inparam['WHAM_TOL'])
    #enter the code for starting running WHAM using class WHAM
    whamrun = WHAM(indata, inparam["NFRAMES"], inparam['U_FORCES'], inparam['U_MEANS'], inparam["TEMP"], inparam['NCVS'], inparam['NUMB'], inparam['GRID_POINTS'], inparam['PERIODICITY'], WHAM_ITERATIONS, inparam['WHAM_TOL'], "Fesout.dat")
    whamrun.start()
    return None

if __name__=="__main__":
    main(inarg)
