#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
#IMPORTS

import numpy as np

from math import *
from matplotlib import pyplot as plt 

#FONCTIONS

def CoordsMoyennes(liste_points):
    
    """
    Fonctions qui renvoi le barycentre d'un nuage de points
    
    :param liste_points: Liste des points du nuages sous la forme [mat , X , Y]
    :type liste_points: array
    
    :return: Coordonnées du barycentre [x , y]
    :rtype: array
    """
    
    n = len(liste_points)
    x = 0
    y = 0 
    for i in range (n):
        x += liste_points[i][1]
        y += liste_points[i][2]
        
    x = x/n
    y = y/n
    
    return [x , y]

def MatriceRefs(b):
    """
    Fonction qui renvoi le vecteur des coordonnées des référence utilisé alignés au mesures b
    
    :param b: Vecteur des mesures
    :type b: array
    
    :param return: Vecteur des coordonnées des références
    :rtype: array
    """
    
    n = len(b)
    refs = np.zeros([n,2])
    for i in range (n):
        refs[i] = [ liste_coord_refs[int(i//(n/len(liste_coord_refs)))][1] , liste_coord_refs[int(i//(n/len(liste_coord_refs)))][2] ]
        
    return refs

def MatriceMesures(liste_mesures):
    """
    Fonctions qui renvoi le vecteur des mesures
    faites au ruban et au double pas
    
    :param liste_mesures: matrice des mesures [point_de_mesure , mesure au ruban (m), mesure au double pas (m)]
    :type liste_mesures: array

    :param return : Vecteur des mesures
    :rtype : array
    """
    n = len(liste_mesures)
    b = np.zeros([2*n,1])
    index = 0
    for k in range (len(liste_coord_refs)):
        mat = liste_coord_refs[k][0]
        for i in range (n):
            if mat == liste_mesures[i][0]:
                b[index][0] = liste_mesures[i][1]
                b[index+1][0] = liste_mesures[i][2]
                index += 2
            
    return b

def MatricePoids(liste_mesures):
    """
    Fonctions qui renvoi le vecteur des poids des mesures
    selon si elle est faites au ruban ou au double pas
    
    :param liste_mesures: matrice des mesures [point_de_mesure , mesure au ruban (m), mesure au double pas (m)]
    :type liste_mesures: array
    
    :param return : Vecteur des poids
    :rtype : array
    """
    n = len(liste_mesures)
    p = np.zeros([2*n,2*n])
    for i in range (n):
        p[2*i][2*i] = poid_mesure_ruban
        p[2*i+1][2*i+1] = poid_mesure_2pas
        
    return p

def MatricePseudoMesure(matrice_mesures , refs):
    """
    Fonction qui renvoi le vecteur des pseudos mesures
    faites depuis le point mat_point
    
    :param matrice_mesures: Vecteur des mesures
    :type matrice_mesures: array
    
    :param return: Vecteur des pseudos mesures
    :rtype: array
    """
    
    n = len(matrice_mesures)
    B = np.zeros([n,1])
    for i in range (n):
        B[i][0] = matrice_mesures[i][0] - sqrt((float(refs[i][0] - M0[0]))**2 + (float(refs[i][1] - M0[1]))**2)
        
    return B


def MatriceModèle(liste_mesures , refs):
    """
    Fonction qui renvoi la matrice modèle A en foncction du point
    de référence mat-point
    
    :param liste_mesures : liste des mesures terrain 
    :type matrice_mesures: array
    
    
    :param return: Matrice du modèle
    :rtype: array
    """
    
    n = len(liste_mesures)
    A = np.zeros([n,2],dtype = 'float')
    
    for i in range (n):
        d = sqrt((float(refs[i][0] - M0[0]))**2 + (float(refs[i][1] - M0[1]))**2)
        A[i][0] = (M0[0] - refs[i][0])/d
        A[i][1] = (M0[1] - refs[i][1])/d
        
    return A


def CalculXEstim(A , P , B):
    """
    Fonction qui renvoi une matrice d'estimation des paramètre X
    
    :param A: Matrice du modèle
    :type A: array
    :param P: Matrice de poids
    :type P: array
    :param B: Matrice des pseudos observations
    :type B: array
    
    :param return: Matrice d'estimation des paramètres
    :rtype : array
    """
    
    X = np.dot(A.transpose() , P)
    X = np.dot(X , A)
    X = np.linalg.inv(X)
    X = np.dot(X , A.transpose())
    X = np.dot(X , B)
    
    return X

def CalculVEstim(B , A , Xestim):
    """
    Fonction qui renvoi une matrice estimé des résidu
    
    :param B: Matrice des mesures
    :type B: array
    :param A: Matrice du modèle
    :type A: array
    :param Xestim: Matrice des pseudos observations
    :type Xestim: array
    
    
    :param return: Matrice d'estimation des résidus
    :rtype : array
    """
    
    return B - np.dot(A , Xestim)

def FacteurUnitaireDeVariance (Vestim , P , B , Xestim):
    """
    Fonction qui renvoi le facteur unitaire de variance des mesures
    
    :param Vestim: Matrice d'estimation des résidus
    :type Vestim : array
    :param P: Matrice de poids
    :type P: array
    :param B: Matrice des mesures
    :type B: array
    :param Xestim: Matrice d'estimation des paramètres
    :type Xestim : array
    
    :param return: Valeur du facteur unitaire de variance
    :rtype: float
    """
    
    n = len(B)
    p = len(Xestim)
    
    sigma = np.dot(Vestim.transpose(),P)
    sigma = np.dot(sigma , Vestim)
    s = sigma[0][0]
    s = s/(n-p)
    
    return s

def VarianceXestim(sigma2 , A , P):
    """
    Fonction qui renvoi la variance de Xestimé
    
    :param sigma2: Valeur du facteur unitaire de variance
    :type sigma2: float
    :param A: Matrice du modèle
    :type A: array
    :param P: Matrice de poids
    :type P: array
    
    :param return: Variance de la matrice paramètre estimé
    :rtype : array
    
    """
    
    var = np.dot(A.transpose() , P)
    var = np.dot(var , A)
    var = sigma2 * np.linalg.inv(var)
    
    return var

def VarianceVestim(sigma2 , P , A):
    """
    Fonction qui renvoi la variance de la matrice des résidu estimé
    
    :param sigma2: Valeur du facteur unitaire de variance
    :type sigma2: float
    :param A: Matrice du modèle
    :type A: array
    :param P: Matrice de poids
    :type P: array
    
    :param return: Variance de la matrice des résidus estimés
    :rtype : array
    """
    
    var = np.dot(A.transpose(),P)
    var = np.dot(var , A)
    var = np.dot(A , np.linalg.inv(var))
    var = np.dot(var , A.transpose())
    var = sigma2*(np.linalg.inv(P) - var)
    
    return var

def VNormalisé(Vestim , VarVestim):
    """
    Fonction qui renvoi la matrice des résidus normalisé
    
    :param Vestim: matrice des résidus estimés
    :type Vestim: array
    :param VarVestim: Variance de la matrice des résidus estimés
    :type VarVestim: array
    
    :return : Matrice des résidus normalisés
    :rtype: array
    """
    
    n = len(Vestim)
    Vnorm = np.zeros([n,1])
    for i in range(n):
        Vnorm[i][0] = Vestim[i][0]/(sqrt(VarVestim[i][i]))
        
    return Vnorm

def XNormalisé(Xestim , VarXestim):
    """
    Fonction qui renvoi la matrice des paramètres normalisé
    
    :param Xestim: matrice des résidus estimés
    :type Xestim: array
    :param VarXestim: Variance de la matrice des résidus estimés
    :type VarXestim: array
    
    :return : Matrice des paramètres normalisés
    :rtype: array
    """
    
    n = len(Xestim)
    Xnorm = np.zeros([n,1])
    for i in range(n):
        Xnorm[i][0] = Xestim[i][0]/(sqrt(VarXestim[i][i]))
        
    return Xnorm

def Linearisation(liste_mesures , epsilon):
    
    """
    Fonction qui renvoi les coordonnées d'un point par estimation
    des moindres carrées
    
    :param liste_mesures: liste des mesures réalisées
    :type liste_mesures: array
    :param epsilon: Seuil de précision de l'estimation (m)
    :type epsilon: float
    
    :param return: Coordonnées calculées du point
    :rtype: array
    """
    b = MatriceMesures(liste_mesures)
    nb_mesures = len(b)
    refs = MatriceRefs(b)
    p = MatricePoids(liste_mesures)
    B = MatricePseudoMesure(b , refs)
    A = MatriceModèle(B , refs)
    Xestim = CalculXEstim(A , p , B)
    Vestim = CalculVEstim(B , A , Xestim)
    s2 = FacteurUnitaireDeVariance(Vestim , p , B , Xestim)
    varXestim = VarianceXestim(s2 , A , p)
    varVestim = VarianceVestim(s2 , p , A)
    Vnorm = VNormalisé(Vestim , varVestim)
    Xnorm = XNormalisé(Xestim , varXestim)
    
    dist = sqrt(Xestim[0]**2 + Xestim[1]**2)

    test = TestRésidusNormalises(Vnorm , b , p ,refs)
    b = test[0]
    p = test[1]
    refs = test[2]
    
    while dist > epsilon :
        M0[0] += Xestim[0][0]
        M0[1] += Xestim[1][0]
        B = MatricePseudoMesure(b , refs)
        A = MatriceModèle(B , refs)
        Xestim = CalculXEstim(A , p , B)
        Vestim = CalculVEstim(B , A , Xestim)
        s2 = FacteurUnitaireDeVariance(Vestim , p , B , Xestim)
        varXestim = VarianceXestim(s2 , A , p)
        varVestim = VarianceVestim(s2 , p , A)
        Vnorm = VNormalisé(Vestim , varVestim)
        Xnorm = XNormalisé(Xestim , varXestim)
        dist = sqrt(Xestim[0]**2 + Xestim[1]**2)
        
        test = TestRésidusNormalises(Vnorm , b , p ,refs)
        b = test[0]
        p = test[1]
        refs = test[2]
        
    print('Nombre de mesures utilisées: '+str(len(B))+'/'+str(nb_mesures))
    histogramme_residus(Vnorm)
    return(M0)
    
def histogramme_residus(Vnorm) : 
    """
    Fonction qui permet l'affichage d'un histogramme des résidus normalisés.
    
    :param matrice_normalise: matrice des residus normalises
    :type matrice_normalise: matrice 
    """
    
    
    plt.hist(Vnorm, range = (-3, 3), bins = 50, color = 'blue',
                edgecolor = 'black')
    plt.xlabel('valeurs')
    plt.ylabel('nombres')
    plt.title('Histogramme des résidus normalisés')
    

def ecrireDansFichier(texte):
    
    """
    Fonction qui écrit dans un fichier texte un texte
    
    :param texte: Texte à ajouter
    :type texte: string
    """
    fichier = open('data_produit/Coordonnées.dat',"w")
    fichier.writelines(texte)
    fichier.close()
    
def MoyenneListe(liste):
    """
    Fonction qui renvoi la moyenne d'une liste de donnée
    
    :param liste: Liste de données
    :type liste: array
    
    :param return: Moyenne de la liste
    :rtype: float
    """
    
    n = len(liste)
    s = 0
    for i in range (n):
        s+= liste[i]
        
    return s/n

def VarianceListe(liste):
    """
    Fonction qui renvoi la variance d'une liste
    
    :param liste: Liste de données
    :type liste: array
    
    :param return: Variance de la liste
    :rtype: float
    """
    
    n = len(liste)
    moy = MoyenneListe(liste)
    V = 0
    for i in range (n):
        V += (liste[i] - moy)**2
        
    return V/n

def TestRésidusNormalises(Vnorm , b , p , refs):
    
    """
    Fonctions qui supprime les valeurs (mesure , poids) dont les résidus normalisé sont
    supérieur en valeur absolue à 3
    
    :param Vnorm: Vecteur des Résidus normalisés
    :type Vnorm: array
    :param b: Vecteur des mesures
    :type b: array
    :param p: Matrice carrée des poids des mesures
    :type p: array
    :param refs: Matrice des références utilisées pour chaque mesure de b
    :type refs: array
    
    :param return: Matrice composé des différents vecteurs et matrices tronqué des valeurs à supprimer: [b , p ,refs]
    :rtype: array
    """
    
    n = len(Vnorm)
    for i in range (n):
        if abs(Vnorm[i]) >= 3:
            b = np.delete(b , i , 0)
            refs = np.delete(refs , i , 0)
            p = np.delete(np.delete(p , i , 0) , i , 1)
    return [b , p , refs]

    


if __name__ == "__main__":
    
 
    
    # VARIABLES
    
    '''récupération des mesures'''
    liste_coord_refs = np.genfromtxt('data/coord_points_reference.dat')
    liste_mesures = np.genfromtxt('data/mesures_2019_1.dat' , delimiter=';')
    
    liste_mesures_2pas = np.genfromtxt('data/mesures_double_pas.dat')
    liste_mesures_ruban = np.genfromtxt('data/mesures_ruban.dat')
    
    '''Affectaion d'un poids à chaque mesure en fonction de son mode d'aquisition'''
    poid_mesure_ruban = 1/VarianceListe(liste_mesures_ruban)
    poid_mesure_2pas = 1/VarianceListe(liste_mesures_2pas)
    
    # ALGO
    
    print('Point mesuré: 43')
    
    M0 = CoordsMoyennes(liste_coord_refs)
    coord = Linearisation(liste_mesures , 0.001)
    print('Coordonnées du point estimées (Lambert 93) : ' + str(coord[0]) + ',' + str(coord[1]))
    ecrireDansFichier('43;'+str(round(coord[0],3))+';'+str(round(coord[1],3)))           
    
  
    print('Réalisé par Charles LAVERDURE & Jules PIERRAT')

