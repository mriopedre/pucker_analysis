#%%
import numpy as np
from tqdm import tqdm
import pandas as pd
import MDAnalysis as md

class PuckerAnalysis:
    # This functions are adapted from Balogh, G. et al. (2021)'s implementation, available at  https://doi.org/10.1021/acs.jcim.1c00200
    
    def __init__(self, u) -> None:
        """
        Initialize with an MDAnalysis Universe object.

        Parameters:
        u (Universe): MDAnalysis Universe containing the trajectory data.
        """
        self.u = u

    def prepare_selections(self, sel: str) -> list:
        """Prepares the selection to be used for subsequent methods

        Args:
            sel (str): MDAnalysis selection string

        Returns:
            list: List of residue IDs
        """
        return list(set(u.select_atoms(sel).resids))

    def cremer_pople_piranose(self, resid: int, ring_atoms: list) -> tuple:
        """
        Calculate Cremer-Pople parameters (Q, theta, phi) for a piranose (6 member) ring in a single frame.

        Parameters:
        resid (int): Residue ID of the ring.
        ring_atoms (list): List of atom names in the ring.

        Returns:
        tuple: Q, theta, and phi values.
        """
        N = len(ring_atoms)  # Determine the number of atoms in the ring; should be 6.
        if N != 6:
            raise ValueError('The ring must have 6 atoms!')  # Ensure the ring has exactly 6 atoms.

        R = np.empty((N, 3))  # Initialize an empty array to store the 3D coordinates of the ring atoms.
        for i, atom in enumerate(ring_atoms):
            # Select the atom in the specified residue by name.
            selection = self.u.select_atoms(f"resid {resid} and name {atom}")
            if not selection:
                raise ValueError(f"Atom '{atom}' in residue '{resid}' not found!")
            R[i] = selection.positions[0]  # Store the coordinates of the selected atom.

        R_cog = np.mean(R, axis=0)  # Compute the center of geometry of the ring atoms.
        R0 = R - R_cog  # Translate the ring atoms so that the center of geometry is at the origin.

        # Calculate R1 and R2 vectors using sinusoidal functions to describe the ring's geometry.
        R1 = np.sum([R0[j] * np.sin(2 * np.pi * j / N) for j in range(N)], axis=0)
        R2 = np.sum([R0[j] * np.cos(2 * np.pi * j / N) for j in range(N)], axis=0)

        n = np.cross(R1, R2)  # Compute the normal vector to the plane defined by R1 and R2.
        n /= np.linalg.norm(n)  # Normalize the normal vector.
        z = np.dot(R0, n)  # Project the translated coordinates onto the normal vector.

        Q = np.sqrt(np.sum(z**2))  # Calculate the puckering amplitude Q.

        # Compute q2cos and q2sin components for the azimuthal angle phi.
        q2cos = np.sqrt(2 / N) * np.sum([z[j] * np.cos(4 * np.pi * j / N) for j in range(N)])
        q2sin = -np.sqrt(2 / N) * np.sum([z[j] * np.sin(4 * np.pi * j / N) for j in range(N)])

        phi = np.degrees(np.arctan2(q2sin, q2cos)) % 360  # Calculate the azimuthal angle phi in degrees.

        q3 = np.sqrt(1 / N) * np.sum([((-1) ** j) * z[j] for j in range(N)])  # Compute q3 for the polar angle theta.
        theta = np.degrees(np.arccos(q3 / Q))  # Calculate the polar angle theta in degrees.

        return Q, theta, phi  # Return the computed Cremer-Pople parameters.

    def cremer_pople_analysis(self, resids: list) -> np.ndarray:
        """
        Perform Cremer-Pople analysis across a trajectory for specified piranose rings.

        Parameters:
        resids (list): List of residue IDs to analyze.

        Returns:
        np.ndarray: Array of Cremer-Pople parameters for each frame and residue.
        """
        ring_atoms = ['O5', 'C1', 'C2', 'C3', 'C4', 'C5']
        cp_data = []

        for ts in tqdm(self.u.trajectory, desc="Processing frames"):
            frame_data = [self.cremer_pople_piranose(resid, ring_atoms) for resid in resids]
            cp_data.append([item for sublist in frame_data for item in sublist])

        return np.array(cp_data)
    
    def make_theta_phi_dict(self, cp):
        """
        Converts the results from cremer_pople_analysis() into a nested dictionary structure.

        Parameters:
        cp (list): Output from cremer_pople_analysis(), a list of Cremer-Pople parameters.

        Returns:
        tuple: A dictionary with the structure {frame: {resid: {'Q': value, 'theta': value, 'phi': value}}}
            and a list of residue IDs.
        """
        resids = list(range(1, len(cp[0]) // 3 + 1))
        return_d = {
            frame: {
                res: {'Q': ts[3 * i], 'theta': ts[3 * i + 1], 'phi': ts[3 * i + 2]}
                for i, res in enumerate(resids)
            }
            for frame, ts in enumerate(cp)
        }
        return return_d, resids

    def _determine_pucker_type(self, theta):
        """
        Determines the pucker conformation based on the theta angle.
        It is very handwaving and only meant to work for Iduronic acid.
        It should be modified for any other pupose.

        Parameters:
        theta (float): The theta angle from Cremer-Pople analysis.

        Returns:
        str: The pucker conformation ('1C4', '2S0', '4C1', or 'NA').
        """
        if theta > 135:
            return '1C4'
        elif 45 <= theta <= 135:
            return '2S0'
        elif theta < 45:
            return '4C1'
        else:
            return 'NA'

    def make_puckers_dict(self, return_d):
        """
        Generates a dictionary mapping each frame and residue to its pucker conformation.

        Parameters:
        return_d (tuple): Output from make_theta_phi_dict(), containing the nested dictionary of Cremer-Pople parameters and the list of residue IDs.

        Returns:
        dict: A dictionary with the structure {frame: {resid: pucker_conformation}}.
        """
        cp_dict, resids = return_d
        puckers = {
            frame: {
                res: self._determine_pucker_type(cp_dict[frame][res]['theta'])
                for res in resids
            }
            for frame in cp_dict
        }
        return puckers

    def present_puckers_as_df(self, puckers):
        """
        Present the puckers dictionary as a DataFrame.
        In horizontal - resid ID
        In vertical - frame number

        Parameters:
        puckers (dict): Dictionary of puckers generated by make_puckers_dict().

        Returns:
        pd.DataFrame: DataFrame with the puckers data.
        """
        return pd.DataFrame(puckers).T

#%%
# Example usage
# Select simulation files and selection of the residues
#fol = '/example_folder/'
fol = '/wrk/gromacs/lumi/metad-patterns/AIDOA-2S/sim/run/'
#fol = '/wrk/gromacs/lumi/metad-patterns/AIDOA-2S/sim/run_4c1/'

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
df = PC.present_puckers_as_df(puckers)

