import streamlit as st
import pandas as pd
from pandas.core.arrays import categorical
import numpy as np
import matplotlib as plt
import plotly.express as px
import matplotlib.pyplot as plt

st.image('BANNIERE MAIL ehlcathol.png')
st.title('Deep-Learning for the estimation of RON, MON and CN')
st.subheader('Method available for any C, H, O, N hydrocarbons')
st.write("----------------------------------------------------------")
 

######################################################################################################
## Premiere partie sur l'exploitation de la database experimentale
######################################################################################################

#st.header("_Experimental Database_")
#
#RON = pd.read_csv('Database_RON_EXP.txt',sep = ',')
#RON = RON.drop(['No.'], axis=1)
#st.dataframe(RON)
#
#st.write("")
#st.markdown('- ASTM, 1958 : "American Society for Testing Materials, Knocking Characteristics of Pure Hydrocarbons, American Petroleum Institute Research Project 45 (1958)"')
#st.markdown('- Law et al., 2017 : "K. Lawyer, Incorporation of higher carbon number alcohols in gasoline blends for application in spark-ignition engines, Michigan Technological University (2017)"')
#st.markdown('- NREL Database : "Co-Optimization of Fuels & Engines (Co-Optima) Initiative NREL Database, accessed in July 2019, https://www.nrel.gov/"')
#st.markdown('- McCormick et al., 2017 : "R.L. McCormick, G. Fioroni, L. Fouts, E. Christensen, J. Yanowitz, E. Polikarpov, K. Albrecht, D.J. Gaspar, J. Gladden, A. George. Selection criteria and screening of potential biomass-derived streams as fuel blendstocks for advanced spark-ignition engines. SAE International Journal of Fuels and Lubricants, 10 (2017) 442-460"')
#st.markdown('- Yanowitz et al., : "2011	J. Yanowitz, E. Christensen, R.L. McCormick, Utilization of Renewable Oxygenates as Gasoline Blending Components, NREL Technical Report TP-5400-50791 (2011)"')
#st.markdown('- Hunwartzen et al., : "1982	I. Hunwartzen, Modification of CFR test engine unit to determine octane numbers of pure alcohols and gasoline-alcohol blends, SAE Technical Paper 820002 (1982)"')
#st.markdown('- ECRL Database : "ECRL Database, https://database.uml-ecrl.org/"')
#st.markdown('- ASTM, 2019 : "ASTM D2699-19 Standard Test Method for Research Octane Number of Spark-Ignition Engine Fuel. American Society for Testing and Materials (ASTM) international; 2019"')
##
#st.write("")
#st.markdown('#### Graphical Representation of the Database') 
#
#
#numerical_cols = RON[['Molecular weight [g/mol]','RON (pure component) experimental']].columns.to_list()
#categorical_cols = RON.select_dtypes(include=['object']).columns.to_list()
#
#var_x = st.selectbox('X-axis', numerical_cols)
#var_y = st.selectbox('Y-axis', numerical_cols)
#var_color = st.selectbox('Colorized family points', categorical_cols)
#
#fig = px.scatter( data_frame=RON,
#                 x = var_x,
#                 y = var_y,
#                 color = var_color,
#                 title = str(var_y)+" VS "+str(var_x)+" by "+str(var_color)
#                )
#st.plotly_chart(fig)


######################################################################################################
## Deuxième partie sur l'estimation du RON à partir du smile de la molécule
######################################################################################################

st.header("_Model based on a QSPR approach_")
st.write("")

import rdkit
import rdkit.Chem.inchi

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw

import mordred
from mordred import Calculator, descriptors

import pandas as pd
import numpy as np

#
## fonction d'appelle pour tous les descripteurs
#
def All_Mordred_descriptors_1(SMILES):
    calc = Calculator(descriptors, ignore_3D=True) # appel aux descripteurs
    n_2D = len(calc.descriptors) # nombre de descripteurs
    mol = Chem.MolFromSmiles(SMILES) # transformation de smile en MOL
    
    if not mol :
        st.write('**Warning !!**') # Dans le cas ou Rdkit n'arrive pas à trouver le graph de la molecule
        st.write('Python argument types in rdkit.Chem.rdmolfiles.MolToSmiles(NoneType) did not match C++ signature')
        st.write('The problem comes from Rdkit package - Try with another molecule')
    
    # On utilise cette façon car avec la méthode panda il cree une barre de progession que bloque Streamlit
    df = calc(mol) #[:n_2D] # calcul de tous les descripteurs
    #df = df.drop_missing().asdict() # tranformation des résultats en un dictionnaire avec les valeurs et entêtes 
    df = df.asdict() # tranformation des résultats en un dictionnaire avec les valeurs et entêtes
    df = pd.DataFrame.from_dict(df, orient ='index') # tranformation du dictionnaire en dataframe
    df = df.T # transposition du dataframe
    return df

#
## fonction qui prend un SMILES (str) et renvoie la notation Inchikey, le SMILES canonique et une liste avec le nombre d'atomes de C,H,O,N
#
def smiles_to_Inchikey_and_molecule_1(SMILES):    
    mol = Chem.MolFromSmiles(SMILES)
    
    if not mol :
        st.write('**Warning !!**') # Dans le cas ou Rdkit n'arrive pas à trouver le graph de la molecule
        st.write('Python argument types in rdkit.Chem.rdmolfiles.MolToSmiles(NoneType) did not match C++ signature')
        st.write('The problem comes from Rdkit package - Try with another molecule')
        #SMILES_Molecules = 'C1C=CC=C(OC)C=1' # Molécule par défaut
        #D2 = smiles_to_Inchikey_and_molecule_1(SMILES_Molecules)
        #return
    
    smiles = Chem.MolToSmiles(mol)
    Inchikey = rdkit.Chem.inchi.MolToInchiKey(mol)
    descriptors = All_Mordred_descriptors_1(SMILES)
    nC = descriptors["nC"].iloc[0]
    nH = descriptors["nH"].iloc[0]
    nO = descriptors["nO"].iloc[0]
    nN = descriptors["nN"].iloc[0]
    return Inchikey,smiles,nC,nH,nO,nN

def smiles_to_Inchikey(SMILES):
    mol = Chem.MolFromSmiles(SMILES)
    smiles = Chem.MolToSmiles(mol)
    Inchikey = rdkit.Chem.inchi.MolToInchiKey(mol)
    return(Inchikey)
  
#
## Enregistrement d'une molécule en notation SMILE
#
SMILES_Molecules = 'C1C=CC=C(OC)C=1' # Molécule par défaut

import streamlit as st
from streamlit_ketcher import st_ketcher


DEFAULT_MOL = SMILES_Molecules
molecule = st.text_input("Insert the SMILE notation of the molecule or draw it", DEFAULT_MOL)
SMILES_Molecules = st_ketcher(molecule)
st.markdown(f"Smile code: ``{SMILES_Molecules}``")

#
# Conversion en InchiKey + Formule Brute
D2 = []
D2 = smiles_to_Inchikey_and_molecule_1(SMILES_Molecules)
SMILES_Molecules = D2[1]
#
# Calcul de tous les descripteurs
#
D1 = []
D1 = All_Mordred_descriptors_1(SMILES_Molecules)
#
st.write('your Inchikey notation is: ', D2[0])
st.write('your canonical smiles notation is: ', D2[1])
st.write('number of Carbon atom -   C: ', D2[2])
st.write('number of Hydrogen atom - H: ', D2[3])
st.write('number of Oxygen atom-    O: ', D2[4])
st.write('number of Nitrogen atom-  N: ', D2[5])
st.write('exact molecular weight [g/mol]:', D1['MW'])

st.write('all the descriptors (for more information about descriptiors see [documentation](http://mordred-descriptor.github.io/documentation/master/descriptors.html))', D1)
#st.dataframe(D1)
st.write('111111111111111111:', D1['ABC'])
st.write('222222222222222222:', D1['ABC'].iloc[0])
dede=3.2
dede=D1['ABC'].iloc[0]
st.write('333333:', dede)
# Transformation en dataframe
D1 = pd.DataFrame(D1)

# Partie Machine Learning - Estimation du RON avec le modèle 4

import joblib
import sklearn
#
# Importation des differents modèles ANN avec métadata
#
model_mordred_457 = joblib.load('./Modele_457.joblib') # Pour l'estimation des RON
model_mordred_457_MON = joblib.load('./Modele_MON_457_1_VF.joblib') # Pour l'estimation des MON
model_mordred_457_CN = joblib.load('./Modele_CN_457_6_VF.joblib') # Pour l'estimation des CN
#
# Je ne vais garder que les 457 trouvés avec le modèle RON
#
# Lire les noms de colonnes d'un fichier texte dans une liste
with open('noms_colonnes_457.txt', 'r') as f:
    liste_espece_457 = [ligne.strip() for ligne in f]
 
# Calcul de Tous Les Descripteurs
X_sim_Model_4 = All_Mordred_descriptors_1(SMILES_Molecules)
# Réduction aux colones uniquement mécessaires
X_sim_Model_4 = X_sim_Model_4[liste_espece_457]
# Nettoyage colone n'ayant pas convergé
X_sim_Model_4 = X_sim_Model_4.astype(float)
# Nettoyage colone n'ayant pas convergé
X_sim_Model_4 = X_sim_Model_4.fillna(0)


# Estimation des RON, MON, CN d'après les modèles ANN
Y_sim_predit_Modele_4 = model_mordred_457.predict(X_sim_Model_4)
Y_sim_predit_Modele_MON = model_mordred_457_MON.predict(X_sim_Model_4)
Y_sim_predit_Modele_CN = model_mordred_457_CN.predict(X_sim_Model_4)
#
#st.write('RON with Mordred descriptors:', Y_sim_predit_Modele_4[0].round(1) , 'at +/- 1 ' )
#st.write('MON with Mordred descriptors:', Y_sim_predit_Modele_MON[0].round(1) , 'at +/- 1 ' )
#st.write('CN with Mordred descriptors:', Y_sim_predit_Modele_CN[0].round(1) , 'at +/- 1 ' )
#
RON_predit_Mordred = Y_sim_predit_Modele_4

#####################################################################################################
# Troisème partie sur l'estimation du RON à partir du smile de la molécule - INCHIKEY
#####################################################################################################

#
import tensorflow
#
import tensorflow as tf
import tensorflow.keras as keras
#
import os,sys,h5py,json
from importlib import reload
#
#from tensorflow.keras import Model
#from tensorflow.keras.layers import Dense, Input,Concatenate,Flatten,Embedding
#from tensorflow.keras.optimizers import Adam
#
#
#
#  Step 3 - J'importe les data info nécessaires au modèle InChiKey
#
DataInfo_model_InChiKey = joblib.load('./DataInfo_InChiKey_Embedding.joblib')
#
#  Step 4 - Je calcule les data nécessaires au modèle InChiKey
#
mean_RON = DataInfo_model_InChiKey['mean_RON']
std_RON = DataInfo_model_InChiKey['std_RON']
#
#RON_normalised = pd.DataFrame((RON-mean_RON)/std_RON)
#
#RON_normalised
#st.write('mean_RON: ', mean_RON )
#st.write('std_RON: ', std_RON )
#
# Step 2 - J'importe le modèle IA InChiKey - Embedding
#
from tensorflow.keras.models import load_model
#
model_InChiKey = load_model('./Model_InChiKey.h5')
#                             
#
# dictionnaire des caracteres - on compte également le tiret '-'
#
Inchi_Cars_dict = {'A': 0,
 'B': 1,
 'C': 2,
 'D': 3,
 'E': 4,
 'F': 5,
 'G': 6,
 'H': 7,
 'I': 8,
 'J': 9,
 'K': 10,
 'L': 11,
 'M': 12,
 'N': 13,
 'O': 14,
 'P': 15,
 'Q': 16,
 'R': 17,
 'S': 18,
 'T': 19,
 'U': 20,
 'V': 21,
 'W': 22,
 'X': 23,
 'Y': 24,
 'Z': 25,
 '-': 26 }

# Takes the 14 first characters of an inchikey and transforms them into a vector with numbers corresponding to the places
# of the letters in the dictionnary Inchi_cars_dict
def vectorization(Inchikey):
    vect = [ Inchi_Cars_dict[w] for w in Inchikey ]
    return(vect)
#
# Step 7 : je passe au deuxième modèle, celui InChiKey - Je passe toute les data en notation InchiKey
#
Inchikeys = []
for n in range(len(SMILES_Molecules)):
    if isinstance(SMILES_Molecules, list):
        Inchikeys.append(smiles_to_Inchikey(SMILES_Molecules[n]))
    else:
        Inchikeys.append(smiles_to_Inchikey(SMILES_Molecules))
Inchikeys = pd.DataFrame(Inchikeys,index = np.arange(0,len(SMILES_Molecules)))
#
# Step 8 : je tokenise toute les notations InChiKey
#
Inchikeys_vect  = []
for x in Inchikeys[0]:
    Inchikeys_vect.append(vectorization(x[:25])) # on prennds les 14 + 10 + tiret = 25
Inchikeys_vect = np.array(Inchikeys_vect)
#
# Step 9 : j'estime les RON avec le modèle InChiKey - Avec ce modèle les RON sont normalisés
#
RON_predit_norm = pd.DataFrame(model_InChiKey.predict(Inchikeys_vect))
#
# Step 10 : je dé-normalise les data
#
RON_predit_InChiKey = RON_predit_norm * std_RON + mean_RON
#
#st.write('RON InChiKey:', RON_predit_InChiKey[0][0].round(1))
#

######################################################################################################
## Fonction pour verifier si on est dans le cas d'une chiralite cis/trans : UHFFFAOYSA
######################################################################################################
#
#  Fonction pour verifier si on est dans le cas d'une chriralite cis/trans : UHFFFAOYSA
#
test=[]
test= ["no" if Inchikeys[0][ind][15:25] == 'UHFFFAOYSA' else "yes" for ind in range(len(Inchikeys_vect))]

st.write('Presence of chirality, isomerism, mesomerism, etc.:',test[0])

#st.write(test[0])
    
######################################################################################################
## Fonction pour calculer le RON final selon Chiralité ou non
######################################################################################################
RON_Final = 0
#
#st.write('RON_Final1:', RON_Final)
#st.write('RON_Final2:', RON_predit_Mordred[0])
#st.write('RON_Final3:', RON_predit_InChiKey[0][0])
#
if test[0] == 'no':
    RON_Final = RON_predit_Mordred[0]
elif abs( RON_predit_Mordred[0] - RON_predit_InChiKey[0][0] ) <= 5:
    RON_Final = 0.5 * RON_predit_Mordred[0] + 0.5 * RON_predit_InChiKey[0][0] 
elif abs( RON_predit_Mordred[0] - RON_predit_InChiKey[0][0] ) <= 10:
    RON_Final = 0.6 * RON_predit_Mordred[0] + 0.4 * RON_predit_InChiKey[0][0]
elif abs( RON_predit_Mordred[0] - RON_predit_InChiKey[0][0] ) <= 20:
    RON_Final = 0.7 * RON_predit_Mordred[0] + 0.3 * RON_predit_InChiKey[0][0]        
else: 
    RON_Final = 0.9 * RON_predit_Mordred[0] + 0.1 * RON_predit_InChiKey[0][0]    

#
st.subheader('Results:')
#
#st.write('RON corrected with the InChiKey correction:', RON_Final.round(1) , 'at +/- 1 ')
st.write('RON:', RON_Final.round(1) , 'at +/- 1 ')
#
#st.write('RON with Mordred descriptors:', Y_sim_predit_Modele_4[0].round(1) , 'at +/- 1 ' )
#
st.write('MON:', Y_sim_predit_Modele_MON[0].round(1) , 'at +/- 1 ' )
#
if SMILES_Molecules == 'C':
    Y_sim_predit_Modele_CN[0]=4.98
if SMILES_Molecules == 'CC':
    Y_sim_predit_Modele_CN[0]=6.79 
 
st.write('CN:', Y_sim_predit_Modele_CN[0].round(1) , 'at +/- 1 ' )

######################################################################################################
## Bloc Tampon - Bout de code
######################################################################################################
#
# Test
#st.write('your 55 notation is:', D1.columns)
#st.write('your 66 notation is:', D1['ABC'])
#texte = ['ABC','ABCGG']
#st.write('your 666 notation is:', D1[texte])
#
#st.write('your XX notation is:', DD)
#st.dataframe(D1)
# Les feastures retenues: st.write('your 66666 notation is:', Model_4_RON_Avril2023['Features'])
# st.write('your 6666 notation is:', X_sim_Model_4)
# st.write('all the descriptors (for more insformation see http://mordred-descriptor.github.io/documentation/master/descriptors.html)', D1)



