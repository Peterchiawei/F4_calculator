import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import copy as cp

# F4 calculation between two oxygen atoms of the water molecule which in the cutoff range(0.35nm)
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


# F4 calculation of one frame
def one_frame_calcu( f, F4_data, time_data):
    print("the calculating frame is",f,"\n")
    # Access information for each frame
    coordinates = cp.copy(u.trajectory[f].positions)
   
    # Initialize the parameters
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
    
    # Output the average F4 of the frame
    ave_F4 = F4_sum / F4_counter
    F4_data[f] = float(ave_F4)  
    time_data[f] = f * dt
    print("frame:",f," time:",time_data[f]," F4:",F4_data[f])

# Load the PDB file
file_name = "md_corrected_test2"
pdb_file = file_name + ".pdb"
u = mda.Universe(pdb_file)

# Setting the parameter
atom_number = len(u.atoms)
frame_number = len(u.trajectory)
dt = cp.copy(u.trajectory.dt)
core_number = 20 # The core number be used in multi-process.
np.savetxt(file_name + "_F4.txt", ["Time(ps)    F4_value"], fmt='%s')

# Access the information for each frame
atom_names = u.atoms.names
residue_names = u.atoms.resnames
H2O_cluster = []

# Record the molecular index 
for i in range(0, atom_number):
    # Select the oxygen atoms of the H2O molecule
    if residue_names[i] == 'SOL' and atom_names[i] == 'OW':
        H2O_cluster.append([i,i+1,i+2])

# Iterate through frames by multi-process
if __name__ == '__main__':
    # Initialize the data storage as the dictionary for multi-process
    # Create the vaccum dictionary
    F4_of_one_frame=mp.Manager().dict()
    time=mp.Manager().dict()    
    
    # Initializing with 0.0 in the correct size 
    for f in range(0,len(u.trajectory)):
        time[f] = 0.0
        F4_of_one_frame[f] = 0.0
    #print(time.values(),F4_of_one_frame.values())
    
    # Generate the parameters to run the multi-process
    polylist = []
    for f in range(0,len(u.trajectory)):
        polylist.append((f, F4_of_one_frame, time))
    mypool = mp.Pool(int(core_number))

    
    # Run multi-process
    mypool.starmap(one_frame_calcu, polylist)
    mypool.close()
    mypool.join()
    
    # Output the F4-time array in the file
    np.savetxt(file_name + "_F4.txt", np.transpose([time.values(), F4_of_one_frame.values()]), fmt='%s')

    # Create a line plot
    plt.plot(time.values(), F4_of_one_frame.values())

    # Add labels and a title
    # plt.xlabel('Time (ns)')
    plt.xlabel('Frames')
    plt.ylabel('F4')
    plt.title('F4-time plot')

    # Display the plot (optional)
    plt.savefig(file_name + "_F4.png")

    
