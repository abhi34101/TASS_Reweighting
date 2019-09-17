# TASS_WS_Reweighting
This repository contains some python scripts for unbiasing TASS (Temperature Accelerated Sliced Sampling) and WS-METAD simulations. Additional features are included as and when possible. Also included is a python code for running WHAM on the output probability distribution files obtained after reweighting.
Please see below for details on features.

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
