import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import copy as cp

# F4 calculation between two oxygen atoms
# i&j is the index of the oxygen atoms of the water molecule which in the cutoff range(0.35nm)
# coordinate is the coordin-ate of the oxygen atoms of the water molecule
def f4( O1, H11, H12, O2, H21, H22):        
    # Determine whose the outermost hydrogen atoms.
    D1 = np.linalg.norm(H11-O2)
    D2 = np.linalg.norm(H12-O2)
    if D1 > D2:
        H_outer1 = H11
    else:
        H_outer1 = H12
    
    D1 = np.linalg.norm(H21-O1)
    D2 = np.linalg.norm(H22-O1)
    if D1 > D2:
        H_outer2 = H21  
    else:
        H_outer2 = H22

    # Get the vector of the O-H and O-O
    vector1 = O1 - H_outer1
    vector2 = O2 - H_outer2
    vector3 = O1 - O2
    
    # Get the normal vector of the plane
    nvector1 = np.cross(vector1, vector3)
    nvector2 = np.cross(vector2, vector3)
    nvector2_transpose = np.transpose(nvector2)

    # Calculate the F4 ( F4 = cos(3theta) )
    cos_theta = 0.000000
    F4 = 0.000000
    cos_theta += np.dot(nvector1, nvector2_transpose) / (np.linalg.norm(nvector1) * np.linalg.norm(nvector2))
    F4 += 4*(cos_theta**3) - 3*cos_theta
    
    return F4


# Load the PDB file
file_name = "md_corrected_test"
pdb_file = file_name + ".pdb"
u = mda.Universe(pdb_file)

# Create a matrix with the coordinates of the atoms to count the bond length
atom_number = len(u.atoms)
np.savetxt(file_name + "_F4.txt", ["Time(ps)    F4_value"], fmt='%s')

# Access the information for each frame
atom_names = u.atoms.names
residue_names = u.atoms.resnames
H2O_cluster = []
skip = 5
nf = u.trajectory.n_frames

# Record the molecular index 
for i in range(0, atom_number):
    # Select the oxygen atoms of the H2O molecule
    if residue_names[i] == 'SOL' and atom_names[i] == 'OW':
        H2O_cluster.append([i,i+1,i+2])

# Iterate through frames
for ts in u.trajectory:
    # Skip frames
    f = cp.copy(ts.frame)
    if f%skip != 0:
        continue
    # Initialize the parameters
    coordinates = u.atoms.positions
    F4_cluster = []
    F4_sum = 0
    F4_counter = 0
        
    # Record the molecular index in the cutoff range(0.35nm = 3.5A)
    for i in range(0, len(H2O_cluster)):
        for j in range(i+1, len(H2O_cluster)):
            # Calculate the distance between two oxygen atoms
            if np.linalg.norm(coordinates[H2O_cluster[i][0]]-coordinates[H2O_cluster[j][0]]) < 3.5:
                F4_cluster.append([H2O_cluster[i][0], H2O_cluster[j][0]])
                   
    # Calculate the F4 between two oxygen atoms
    for k in range(0, len(F4_cluster)):
        i = F4_cluster[k][0]
        j = F4_cluster[k][1]
        F4_sum += (f4(  coordinates[i], coordinates[i+1], coordinates[i+2], coordinates[j], coordinates[j+1], coordinates[j+2]))
        F4_counter += 1
    
    # get the average F4 of the frame
    ave_F4 = round(F4_sum / F4_counter, 4) 
    
    # output F4 of each frame
    print(ts.time, ave_F4)
    
    # Output the F4-time array in the file
    record = "{:.2f}    {:+.4f} \n".format(ts.time, ave_F4)
    with open(file_name + "_F4.txt", 'a') as f:
        f.write(record)



# Open the file in read mode
with open(file_name + "_F4.txt", 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Initialize empty arrays to store data
frames = []
f4_values = []

# Skip the first line (header)
lines = lines[1:]

# Iterate over each line and extract values
for line in lines:
    # Split the line into columns
    columns = line.split()

    # Extract values for Frame and F4_value
    frame = float(columns[0])
    f4_value = float(columns[1])

    # Append values to the arrays
    frames.append(frame)
    f4_values.append(f4_value)

# Create a line plot
plt.plot(frames, f4_values)

# Add labels and a title
# plt.xlabel('Time (ns)')
plt.xlabel('Time(ps)')
plt.ylabel('F4')
plt.title('F4-time plot')

# Display the plot (optional)
# plt.show()
plt.savefig(file_name + "_F4.png")
