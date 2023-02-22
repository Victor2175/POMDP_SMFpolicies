#!/usr/bin/env python3

import os
import sys
import numpy as np

pomdp_class = __import__('POMDP_class')


# define the matrices based on the pomdp class 
def set_matrices(pomdp):
    nbS = len(pomdp.states)
    nbO = len(pomdp.observations)
    nbA = len(pomdp.actions)
    P_init = np.zeros(nbS)
    P_trans = np.zeros((nbS,nbA,nbS))
    P_emis = np.zeros((nbA,nbS,nbO))
    reward = np.zeros((nbS,nbA,nbS))

    for key, value in pomdp.R.items():
        if (key[0]==None) and (key[1]!=None) and (key[2]!=None):
            reward[key[1],:,key[2]] = value
        elif (key[0]!=None) and (key[1]==None) and (key[2]!=None):
            reward[:,key[0],key[2]] = value
        elif (key[0]!=None) and (key[1]!=None) and (key[2]==None):
            reward[key[1],key[0],:] = value
        elif (key[0]!=None) and (key[1]==None) and (key[2]==None):
            reward[:,key[0],:] = value
        elif (key[0]==None) and (key[1]==None) and (key[2]!=None):
            reward[:,:,key[2]] = value
        elif (key[0]==None) and (key[1]!=None) and (key[2]==None):
            reward[key[1],:,:] = value
        elif (key[0]==None) and (key[1]==None) and (key[2]==None):
            reward[:,:,:] = value
        else:
            reward[key[1],key[0],key[2]] = value

        
    for key, value in pomdp.T.items():
        if (key[0]==None) and (key[1]!=None) and (key[2]!=None):
            P_trans[key[1],:,key[2]] = value
        elif (key[0]!=None) and (key[1]==None) and (key[2]!=None):
            P_trans[:,key[0],key[2]] = value
        elif (key[0]!=None) and (key[1]!=None) and (key[2]==None):
            P_trans[key[1],key[0],:] = value
        elif (key[0]==None) and (key[1]==None) and (key[2]!=None):
            P_trans[:,:,key[2]] = value
        elif (key[0]!=None) and (key[1]==None) and (key[2]==None):
            P_trans[:,key[0],:] = value
        elif (key[0]==None) and (key[1]==None) and (key[2]!=None):
            P_trans[:,:,key[2]] = value
        elif (key[0]==None) and (key[1]!=None) and (key[2]==None):
            P_trans[key[1],:,:] = value
        elif (key[0]==None) and (key[1]==None) and (key[2]==None):
            P_trans[:,:,:] = value
        else:
            P_trans[key[1],key[0],key[2]] = value

        
    for key, value in pomdp.Z.items():
        if (key[0]==None) and (key[1]!=None) and (key[2]!=None):
            P_emis[:,key[1],key[2]] = value
        elif (key[0]!=None) and (key[1]==None) and (key[2]!=None):
            P_emis[key[0],:,key[2]] = value
        elif (key[0]!=None) and (key[1]!=None) and (key[2]==None):
            P_emis[key[0],key[1],:] = value
        elif (key[0]==None) and (key[1]==None) and (key[2]!=None):
            P_emis[:,:,key[2]] = value
        elif (key[0]!=None) and (key[1]==None) and (key[2]==None):
            P_emis[key[0],:,:] = value
        elif (key[0]==None) and (key[1]==None) and (key[2]!=None):
            P_emis[:,:,key[2]] = value
        elif (key[0]==None) and (key[1]!=None) and (key[2]==None):
            P_emis[:,key[1],:] = value
        elif (key[0]==None) and (key[1]==None) and (key[2]==None):
            P_emis[:,:,:] = value
        else:
            P_emis[key[0],key[1],key[2]] = value
    
    for key, value in pomdp.start.items():
        P_init[key] = value
                
    return nbS,nbO,nbA,P_init,P_emis,P_trans,reward

# create a folder in a directory if it doen's exist yet
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

# load filename.POMDP and create the corresponding filename.dat file that will be loaded by the julia program.
def create_POMDP(filename):
    pomdp = pomdp_class.POMDPEnvironment('instances_POMDP/{}.POMDP'.format(filename))
    nbS,nbO,nbA,P_init,P_emis,P_trans,Reward = set_matrices(pomdp)
    createFolder('instances')
    if not os.path.exists('instances/{}.dat'.format(filename)):
        with open('instances/{}.dat'.format(filename), 'w+') as the_file:
        
            the_file.write("nbS : " + str(nbS) + "\n")
            the_file.write("nbO : " + str(nbO) + "\n")
            the_file.write("nbA : " + str(nbA) + "\n")
        
            the_file.write("P_init \n")    
            for s in range(nbS):
                the_file.write(str(P_init[s])+ " ")
            the_file.write("\n")

            the_file.write("P_trans \n")
        
            for s in range(nbS):
                for a in range(nbA):
                    for ss in range(nbS):
                        the_file.write(str(P_trans[s][a][ss]) + " ")
                    the_file.write("\n")
                the_file.write("\n")
            the_file.write("\n")

            the_file.write("P_emis \n")
            for a in range(nbA):  
                for s in range(nbS):
                    for o in range(nbO):
                        the_file.write(str(P_emis[a][s][o]) + " ")
                    the_file.write("\n")
                the_file.write("\n")
            the_file.write("\n")
        

            the_file.write("Reward \n")
            for s in range(nbS):
                for a in range(nbA):
                    for ss in range(nbS):
                        the_file.write(str(Reward[s][a][ss]) + " ")
                    the_file.write("\n")
                the_file.write("\n")

    return P_init, P_trans, P_emis, Reward

# load the filename
filename = sys.argv[1]

# write the filename.dat corresponding to the .POMDP file.
p_init, p_trans, p_emis, reward = create_POMDP(filename)
