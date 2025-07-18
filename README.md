NMR Scripts to plot CSPs or intensity losses from titrations, calculate Kd values from titrations,  plot relaxation data, plot HDexchange, etc.

Scripts are only compatible with output .csv files from CCPN v3. See 'https://ccpn.ac.uk/wp-content/uploads/2024/11/CSPTutorial.pdf' to understand how to prepare the .csv file.

Briefly:
- follow all steps to pick and assign peaks in each HSQC spectrum
- create a spectrum group (SG)
- populate the values with relevant information,
  - CSP/IL plots and Kd calculation use titrant conc per spectrum (0, 15, 30, 60, 120, etc) and global value as labelled protein conc (30)
  - HDexchange use times per spectrum (100, 200, 300, 400, etc obtained from readtimes.py) and global arbitrary value (1)
  - T1 and T1rho use relaxation delay per spectrum (1, 20, 40, 60, etc) and global arbitrary value (1)
  - CPMG use echo number per spectrum (1, 3, 5, 8, etc) and global value as fixed tau delay (40)
  - HetNOE use spectrum number (0, 1) and global arbitrary value as (1)
- Move SG into new chemical shift lists
- open 'CM' module
- set up the titration input data and create input data table following the instructions
- export the input data table as .csv (correct format will have two entries per residue for both 1H and 15N)
  - see uploaded .csv files for examples
  - eg. if you have a different number of scans for each experiment, add a new column called scans and populate with the values per spectrum
- To get the baseline noise for CPMG and HetNOE data, zoom into a region in the spectrum without any peaks, right click estimate noise, and manually add baseline noise estimates to the 'signalToNoiseRatio' column per spectrum (I know it's not actually signal to noise but I did not want to add a new column)

Some scripts can use .txt and .ss2 files to overlay the protein sequence and secondary structure onto each plot. Type 'python SCRIPTNAME.py -h' to show the flags for each script and how to customise each plot.
