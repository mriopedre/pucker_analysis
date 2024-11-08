# pucker_analysis

Python code to calculate puckering of monosaccharides in an MD trajectory. 
Specially tuned for charmm36m forcefield and iduronic acid monosaccharides, but adjustable to any use.

Some of the functions are inspired and adapted from Balogh, G. et al. (2021)'s implementation, available at  https://doi.org/10.1021/acs.jcim.1c00200

# WARNING - the trajectory should be made whole before analysis (gmx editconf -pbc mol)

#Example usage:

```python

# Select simulation files and selection of the residues
fol = '/example_folder/'
u = md.Universe(fol + 'nowat.gro', fol + 'whole.xtc')
sel = 'resname AIDOA'

# Initialize the class
PC = PuckerAnalysis(u)

# Prepare the selections as a list of residue IDs
selections = PC.prepare_selections(sel)

# Perform the Cremer-Pople analysis - returns a numpy array with the Q, theta, and phi values
puck = PC.cremer_pople_analysis(selections)

# Convert the results into a nested dictionary structure and show them as a DataFrame
q_tetha_phi = PC.make_theta_phi_dict(puck)
# Puckers dict converts the theta values into puckering types, only usefull for a handwaving classification of IdoA
puckers = PC.make_puckers_dict(q_tetha_phi)
PC.present_puckers_as_df(puckers)
```

