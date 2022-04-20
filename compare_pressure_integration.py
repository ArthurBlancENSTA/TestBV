# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:29:03 2022

@author: arblanc

L'objectif de ce script est de comparer les valeurs de Masse Ajoutée et
d'Amortissement Ajouté et de Forces d'Excitation obtenues en intégrant
manuellement sous Python les données de potentiel obtenues via HydroStar
avec celles directement fournies par HydroStar
"""

from Snoopy import Spectral as sp
import Snoopy.Meshing as msh
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from pyHstar import PrsFile


plt.close('all')

"""
1. Lecture du maillage et extraction des donnees pertinentes pour l'intégration
des potentiels sur la surface de carène
"""

path_hst = r'C:\Users\arblanc\Documents\Formation\Hydrostar\Test_DS\DS_1_20_20\ds_1_20_20.hst'

mesh = msh.HydroStarMesh(path_hst)
hull_mesh = mesh.getHullMesh(0)

hull_mesh.setRefPoint(hull_mesh.integrate_cob()) #pour le calcul des Ma et Ba, on se place par rapport au CoB et non par rapport au pt de ref originel ([0,0,0])

print('cob', hull_mesh.integrate_cob())

gp = hull_mesh.getGaussPoints()
area = hull_mesh.getGaussWiWjdetJ()
normals = hull_mesh.getNormals()
gp_normals = hull_mesh.getNormalsAtGaussPoints() #extraction des normales généralisées (avec ici GM = RefpointM)


rho = 1025  # comment extraire la pression d'ailleurs ?
grav = 9.81

"""
2. Ecriture des points de Gauss dans le fichier hsprs
"""
ar_points = gp
rdfName = r'C:\Users\arblanc\Documents\Formation\Hydrostar\Test_DS\DS_1_20_20\hdf\hsrdf_ds_1_20_20.rd2.h5'
projName = "DS_1_20_20"
out_dir = r"C:\Users\arblanc\Documents\Formation\Hydrostar\Test_DS\DS_1_20_20\data"
prsfilename = "DS_1_20_20.prs"


prs_file = PrsFile(pointsVect=ar_points, rdfname=rdfName,
                   PJName=projName, wdir=out_dir)
prs_file.write(filename=prsfilename)


"""
2. bis Executer le .prs --> effectué dans HydroStar (ICE)
NB : probleme --> non-fonctionnel, corrigé : suppression dans l'en-tete de l'information FILENAME
"""
pot_file = r'C:\Users\arblanc\Documents\Formation\Hydrostar\Test_DS\DS_1_20_20\hdf\p_ds_1_20_20.h5_0000'
pot_db = h5py.File(pot_file)


"""
3. Lire le .hdf de hsprs et intégrer la pression
"""
pressure_file = r'C:\Users\arblanc\Documents\Formation\Hydrostar\Test_DS\DS_1_20_20\hdf\p_ds_1_20_20.h5'
prs_db = h5py.File(pressure_file)

Freq_rad = prs_db["Frequency"][:]

inc_pressure = prs_db['Pressure_Inc_Re'][:] + 1j*prs_db['Pressure_Inc_Im'][:]
# Intd_inc_pressure = (gp[:, -1]*normals[:, -1]*area*inc_pressure).sum(axis=2)
# Intd_inc_ressure = Intd_inc_pressure.reshape(tuple([1]+list(Intd_inc_pressure.shape)))

dif_pressure = prs_db['Pressure_Dif_Re'][:] + 1j*prs_db['Pressure_Dif_Im'][:]
# Intd_dif_pressure = (gp[:, -1]*normals[:, -1]*area*dif_pressure).sum(axis=2)
# Intd_dif_ressure = Intd_dif_pressure.reshape(tuple([1]+list(Intd_dif_pressure.shape)))

# attention, le format de pression n'est plus le même (cf 1 variable pour les 6 DOF) --> (heading, freq, NbPanel, DOF)
rad_pressure = prs_db['Pressure_Rad_Re'][:] + 1j*prs_db['Pressure_Rad_Im'][:]

A = np.zeros([rad_pressure.shape[0], 6, 6])
B = np.zeros([rad_pressure.shape[0], 6, 6])
F = np.zeros([rad_pressure.shape[0], 6], dtype=complex)


def transform_AT_to_A0(AT, x0, y0, z0, xT, yT, zT):
    """
    Permet de déplacer les matrices de raideur et d'amortissement entre un point T et un point M0,
    D'après ce qui a été présenté par Xiao-Bo CHEN dans Hydrodynamics in Offshore and Naval Applications - Part I (p 24, formule 59x)
    """
    A = np.zeros(AT.shape)
    for i in range(AT.shape[0]):
        AT11 = AT[i, 0:3, 0:3]
        AT12 = AT[i, 0:3, 3:6]
        AT21 = AT[i, 3:6, 0:3]
        AT22 = AT[i, 3:6, 3:6]
        
        V = np.array( [ [0, -zT+z0 ,yT-y0] , [zT-z0,0, -xT+x0],  [-yT+y0, xT-x0, 0] ])
        
        A[i, 0:3, 0:3] = AT11
        A[i, 0:3, 3:6] = AT12 + np.dot(V, AT11)
        A[i, 3:6, 0:3] = AT21 + np.dot(V, AT11)
        A[i, 3:6, 3:6] = AT22 + np.dot(V, AT12) - np.dot(V, AT21) - np.dot(np.dot(V, AT11), V)
        
    return(A)


for i_w in range(len(Freq_rad)):
    for i in range(6):
        F[i_w, i] = -1*rho * grav * \
            ((inc_pressure[i_w, 0, :]+dif_pressure[i_w, 0, :])
             * gp_normals[:, i]*area).sum(axis=-1)
        for j in range(6):
            A[i_w, i, j] = rho*grav / Freq_rad[i_w] * \
                (rad_pressure[i_w, :, j, :].imag *
                 gp_normals[:, i]*area).sum(axis=-1)
            B[i_w, i, j] = rho*grav * \
                (rad_pressure[i_w, :, j, :].real *
                 gp_normals[:, i]*area).sum(axis=-1)

cobPoint = hull_mesh.integrate_cob()
refPoint = hull_mesh.getRefPoint()

A = transform_AT_to_A0(A, cobPoint[0],cobPoint[1],cobPoint[2],refPoint[0],refPoint[1],refPoint[2])

B = transform_AT_to_A0(B, cobPoint[0],cobPoint[1],cobPoint[2],refPoint[0],refPoint[1],refPoint[2])

"""
3. Comparaison aux données issues d'HS
"""
list_ids_data = ["SurgeAddedMass", "HeaveAddedMass", "RollAddedMass", "YawAddedMass", "SurgeDamping", "HeaveDamping", "RollDamping", "YawDamping", "Excitation X", "Excitation Y", "Excitation Z", "Excitation MX", "Excitation MY", "Excitation MZ"]
dict_loc_data = {"SurgeAddedMass":  [0, 0], "HeaveAddedMass": [2, 2], "RollAddedMass" : [4,4], "YawAddedMass" : [5,5],"SurgeDamping": [0, 0], "HeaveDamping": [2, 2], "RollDamping" : [4,4], "YawDamping" : [5,5], "Excitation X" : [0], "Excitation Y" : [1], "Excitation Z" : [2], "Excitation MX" : [3], "Excitation MY" : [4], "Excitation MZ" : [5]}
dict_mat_data = {"SurgeAddedMass":  A, "HeaveAddedMass": A, "RollAddedMass" : A, "YawAddedMass" : A,"SurgeDamping": B, "HeaveDamping": B, "RollDamping" : B, "YawDamping" : B, "Excitation X" : F, "Excitation Y" : F, "Excitation Z" : F, "Excitation MX" : F, "Excitation MY" : F, "Excitation MZ" : F}
dict_type_data = {"SurgeAddedMass" : "mass", "HeaveAddedMass": "mass", "RollAddedMass": "mass", "YawAddedMass": "mass", "SurgeDamping": "damping", "HeaveDamping": "damping", "RollDamping": "damping", "YawDamping": "damping", "Excitation X" : "force", "Excitation Y" : "force", "Excitation Z" : "force", "Excitation MX" : "force", "Excitation MY" : "force", "Excitation MZ" : "force"}


# import des données des RAOs calculées
# on paramétrise l'import des données
dict_data_filename = {"SurgeAddedMass": r'\AddedMass_11_11.rao', "HeaveAddedMass": r'\AddedMass_11_33.rao', "RollAddedMass" : r'\AddedMass_11_44.rao', "YawAddedMass" : r'\AddedMass_11_66.rao',"SurgeDamping": r'\WaveDamping_11_11.rao', "HeaveDamping": r'\WaveDamping_11_33.rao', "RollDamping" : r'\WaveDamping_11_44.rao', "YawDamping" :  r'\WaveDamping_11_66.rao',"Excitation X" : r'\Excitation_1_1.rao', "Excitation Y" : r'\Excitation_1_2.rao', "Excitation Z" : r'\Excitation_1_3.rao', "Excitation MX" : r'\Excitation_1_4.rao', "Excitation MY" : r'\Excitation_1_5.rao', "Excitation MZ" : r'\Excitation_1_6.rao'}
main_file = r"C:\Users\arblanc\Documents\Formation\Hydrostar\Test_DS"

# on construit un dictionnaire qui accueillera les données de masse ajoutée et amortissement ajouté issus d'HS
list_ids_MaBa_HS = ["20_20"]
dict_MaBa_HS = {"10_10": {"adress": r'\DS_1_10_10\tmp\rdfCoefs'}, "20_20": {"adress": r'\DS_1_20_20\tmp\rdfCoefs'}, "50_50": {"adress": r'\DS_1_20_20\tmp\rdfCoefs'}}

dict_of_rao = {}
for id in list_ids_MaBa_HS:
    for id_data in list_ids_data:
        fadress = main_file + \
            dict_MaBa_HS[id]["adress"]+dict_data_filename[id_data]
        rao = sp.Rao(fadress)
        dict_MaBa_HS[id][id_data] = rao

# on construit un dictionnaire qui accueillera les données de masse ajoutée et amortissement ajouté issus de l'intégration de pression
dict_MaBa_IntgPres= {}
for id_data in list_ids_data:
    file_out = out_dir + dict_data_filename[id_data]
    if len(dict_loc_data[id_data])==1:
        rao2 = sp.Rao(b=np.array([0.]), w=Freq_rad, cvalue=dict_mat_data[id_data][:, dict_loc_data[id_data]
                  [0]].reshape(1, -1, 1), **sp.Rao.getMetaData(rao))     
    elif len(dict_loc_data[id_data])==2:
        rao2 = sp.Rao(b=np.array([0.]), w=Freq_rad, cvalue=dict_mat_data[id_data][:, dict_loc_data[id_data]
                  [0], dict_loc_data[id_data][1]].reshape(1, -1, 1), **sp.Rao.getMetaData(rao))
    dict_MaBa_IntgPres[id_data]= rao2
    rao2.write(file_out)

# on trace les Ma, Ba pour comparaison
for id_data in list_ids_data:
    fig, ax = plt.subplots()
    dict_MaBa_IntgPres[id_data].plot(ax=ax,  marker="+")
    label = ["Integrated Pressure"]
    for id_mesh in list_ids_MaBa_HS:
        dict_MaBa_HS[id_mesh][id_data].plot(ax=ax, marker="None")
    plt.title(id_data)
    plt.legend(["Integrated Pressure"] + list_ids_MaBa_HS)
    plt.savefig(main_file+r'\figures_PressureInt\_'+id_data+'.png', format='png', dpi=500)

plt.close("all")

for id_data in list_ids_data:
    fig, ax = plt.subplots()
    plt.plot(dict_MaBa_IntgPres[id_data].freq, dict_MaBa_IntgPres[id_data].cvalues.imag.reshape(dict_MaBa_IntgPres[id_data].freq.shape), marker="+")
    label = ["Integrated Pressure"]
    for id_mesh in list_ids_MaBa_HS:
            plt.plot(dict_MaBa_HS[id_mesh][id_data].freq, dict_MaBa_HS[id_mesh][id_data].cvalues.imag.reshape(dict_MaBa_HS[id_mesh][id_data].freq.shape), marker="+")
    plt.title(id_data)
    plt.ylabel(id_data+" imaginary part")
    plt.legend(["Integrated Pressure"] + list_ids_MaBa_HS)
    plt.savefig(main_file+r'\figures_PressureInt\_imag_'+id_data+'.png', format='png', dpi=500)

plt.close("all")

#on automatise un peu !
def adimension(rao, data_type, mass):
    """
    Adimensionne les valeurs de sortie d'une .rao comportant des données de Masse Ajoutée ou d'Amortissement ajouté
    
    Entrée :
        rao = variable portant la rao (class rao)
        data_type = "damping" ou "mass"
        mass = masse de l'objet
    """
    if data_type == "damping":
        div_vect = rao.freq * mass
    elif data_type == "mass":
        div_vect = mass * np.ones(rao.freq.shape)
    else:
        raise NameError("Mauvais Argument de type de données")
    rao2=sp.Rao(b=rao.head, w=rao.freq, cvalue = rao.cvalues / div_vect, **sp.Rao.getMetaData(rao) )
    return(rao2)

def compare_plot(rao1, rao2, label1=None, label2=None, marker1="+", marker2="None", data_name=None, save_file_adress=None):
    fig, ax = plt.subplots()
    rao1.plot(ax=ax,  marker=marker1)
    rao2.plot(ax=ax, marker=marker2)
    if not data_name == None:
        plt.title(id_data)
    if not (label1==None or label2==None):
        plt.legend([label1, label2])
    if not save_file_adress == None:
        plt.savefig(save_file_adress+'.png', format='png', dpi=500)
    plt.plot()
    
def compare_plot_phasis(rao1, rao2, label1=None, label2=None, marker1="+", marker2="None", data_name=None, save_file_adress=None):
    fig, ax = plt.subplots()
    rao2.plot(ax=ax, marker=marker2, part="phasis")
    rao1.plot(ax=ax,  marker=marker1, part="phasis")
    if not data_name == None:
        plt.title(id_data)
        plt.ylabel(id_data)
    if not (label1==None or label2==None):
        plt.legend([label1, label2])
    if not save_file_adress == None:
        plt.savefig(save_file_adress+'.png', format='png', dpi=500)
    plt.plot()


# on trace les Ma, Ba ADIMENSIONNEE pour comparaison
for id_data in list_ids_data[:8]:
    data_type = dict_type_data[id_data]
    compare_plot(adimension(dict_MaBa_IntgPres[id_data], data_type, 2/3*math.pi*rho),adimension(dict_MaBa_HS["20_20"][id_data], data_type, 2/3*math.pi*rho), label1 = "Integrated Pressure", label2 = "HydroStar", data_name=id_data, save_file_adress=main_file+r'\figures_PressureInt_Adim\_'+id_data)
    plt.close('all')

# on trace les Ma, Ba en phase pour comparaison
for id_data in list_ids_data[:8]:
    data_type = dict_type_data[id_data]
    compare_plot_phasis(dict_MaBa_IntgPres[id_data], dict_MaBa_HS["20_20"][id_data], label1 = "Integrated Pressure", label2 = "HydroStar", data_name=id_data, save_file_adress=main_file+r'\figures_PressureInt\_phasis'+id_data)
    plt.close('all')


    

###A FAIRE : trouver le souci pour le Roll et pour les excitations pour les mouvements de rotations --> revoir les normales généralisées ?

"""
4. Transfert de la pression et de la masse ajoutée (CoB --> CoG)
"""