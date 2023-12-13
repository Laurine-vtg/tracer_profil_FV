import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image


# Titre de l'application
# Utilisez st.markdown pour ajouter un titre souligné
st.markdown("<h1 style='text-decoration: underline;'>Tracer un profil F-V</h1>", unsafe_allow_html=True)

# Widgets pour les paramètres physiques
st.write('<span style="color: green;">Complétez les paramètres  ci-dessous.</span>', unsafe_allow_html=True)
nom = st.text_input("Nom Prénom")
num_sprint = st.number_input("Numéro du sprint", min_value=1,max_value=2)
date_du_test = st.text_input("Date du test (jj/mm/aaaa)")
date_de_naissance = st.text_input("Date de naissance (jj/mm/aaaa)")
equipe = st.selectbox("Equipe",["","Réserve","Formation","Avenir"],)
poste = st.selectbox("Poste",["","G","Def","Mil","Att"])
statut = st.text_input("Statut")
masse = st.number_input("Masse (kg)",value=0)
taille_cm = st.number_input("Taille (cm)",value=0)
taille = taille_cm/100
#st.write(taille)
temperature = st.number_input("Température (°C)",value=0)
Pb = st.number_input("Pression atmosphérique (mmHg)",value=0)

rho = 1.293*Pb/760*273/(273+temperature)
Af = (0.2025*(taille**0.725)*(masse**0.425))*(0.266)
Cd = 0.9
k = 0.5*rho*Af*Cd

#st.write("Rho:",rho)
#st.write("Af:",Af)
#st.write("Cd:",Cd)
#st.write("k:",k)

st.write('<span style="color: green;">Choisissez un fichier csv puis sélectionnez votre sprint avec les outils (sélection approximative puis sélection précise) qui sont dans la marge de gauche (pour ouvrir la marge cliquez sur la flèche en haut à gauche).</span>', unsafe_allow_html=True)
# Charger les données à partir du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, delimiter=';', decimal=',')
else:
    st.warning("Veuillez charger un fichier CSV.")


# Indices de lignes par défaut pour la sélection grossière
#x = 50000
#y = x + 10000


# Créer un dictionnaire session_state s'il n'existe pas
if 'coarse_start_index' not in st.session_state:
    st.session_state.coarse_start_index = 0
    st.session_state.coarse_end_index = len(data) - 1

# Définir une fonction pour mettre à jour le graphique
def update_plot(start_index, end_index):
    selected_data = data.iloc[start_index:end_index + 1]

    # Créer le graphique avec la plage sélectionnée
    fig, ax = plt.subplots()
    ax.plot(selected_data.index, selected_data['Velocity'], color="blue")
    ax.set_xlabel('Numéro de ligne')
    ax.set_ylabel('Vitesse')
    #ax.legend()

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

# Utiliser st.sidebar pour afficher les outils de sélection approximative
with st.sidebar:
    st.subheader("Sélection approximative du sprint :")
    coarse_start_index = st.number_input('Numéro de ligne de début', min_value=0, max_value=len(data) - 1, value=0)
    coarse_end_index = st.number_input('Numéro de ligne de fin', min_value=coarse_start_index, max_value=len(data) - 1, value=len(data) - 1)
    
#with st.sidebar.container():
    image = Image.open('photo1.png')
    st.image(image, use_column_width=True, caption = 'Exemple de sélection approximative du sprint')

# Afficher les données sélectionnées grossièrement
#st.write("Sélection grossière:")
#st.write(data.iloc[coarse_start_index:coarse_end_index + 1])

# Utiliser un deuxième slider pour une sélection fine à l'intérieur de la sélection grossière
with st.sidebar:
    st.header("Sélection précise du sprint :")
    fine_range_selector = st.slider('Numéros de ligne', min_value=coarse_start_index, max_value=coarse_end_index,
                                    value=(coarse_start_index, coarse_end_index))
    
#with st.sidebar.container():
    image1 = Image.open('photo2.png')
    st.image(image1, use_column_width=True, caption = 'Exemple de sélection précise du sprint')

st.sidebar.write('<span style="color: green;">Une fois que votre sprint ressemble à celui en exemple juste au dessus, vous pouvez valider votre sprint.</span>', unsafe_allow_html=True)

#selection fine = selection précise et selection grossière = selection approximative
# Extraire la plage de sélection fine
fine_start_index, fine_end_index = fine_range_selector

# Mettre à jour le graphique avec la sélection fine
update_plot(fine_start_index, fine_end_index)

# Afficher les données sélectionnées finement
#st.write("Sélection fine:")
#st.write(data.iloc[fine_start_index:fine_end_index + 1])


# Utiliser st.sidebar.button pour afficher le bouton sur la barre latérale
if st.sidebar.button('Valider le sprint'):
    # Filtrer les données en fonction de la sélection fine
    selected_data = data.iloc[fine_start_index:fine_end_index + 1]

    # Conserver uniquement les colonnes 'Seconds' et 'Velocity'
    selected_data = selected_data.loc[:, ['Seconds', 'Velocity', 'HDOP', ' #Sats']]

    # Normaliser la colonne 'Seconds' pour qu'elle commence à 0
    selected_data['Seconds'] = selected_data['Seconds'] - selected_data['Seconds'].min()

    # Afficher les données sélectionnées
    #st.write("Données sélectionnées:")
    #st.write(selected_data)

    # Créer un nouveau DataFrame avec les données sélectionnées
    new_dataframe = selected_data.copy()

# Renommer la colonne 'Velocity' en 'Vitesse_Reelle' (par exemple)
    new_dataframe = new_dataframe.rename(columns={'Velocity': 'Vitesse_Reelle'})

    # Définir la fonction de modèle
    def modele(params, temps):
        Vmax, Tau, Delay = params
        return Vmax * (1 - np.exp(-(temps - Delay) / Tau))

    # Définir la fonction d'erreur (carré de la différence)
    def erreur(params, temps, vitesse_reelle):
        vitesse_modelee = modele(params, temps)
        square_diff = (vitesse_reelle - vitesse_modelee) ** 2
        return np.sum(square_diff)

    # Initialiser les paramètres
    params_initiaux = [5, 1, 0.1]

    # Optimiser les paramètres pour minimiser l'erreur
    resultat_optimisation = minimize(erreur, params_initiaux, args=(new_dataframe['Seconds'], new_dataframe['Vitesse_Reelle']), method='L-BFGS-B')

    # Obtenir les paramètres optimisés
    params_optimises = resultat_optimisation.x

   # Ajouter la colonne de vitesse modélisée au nouveau DataFrame
    new_dataframe['vitesse_model'] = modele(params_optimises, new_dataframe['Seconds'])

    # Ajouter la colonne de square difference au nouveau DataFrame
    new_dataframe['square_difference'] = (new_dataframe['Vitesse_Reelle'] - new_dataframe['vitesse_model']) ** 2

    # Calculer la somme de la colonne square difference pour le nouveau DataFrame
    somme_square_difference = new_dataframe['square_difference'].sum()
    
# Réinitialiser l'index du DataFrame
    new_dataframe.reset_index(drop=True, inplace=True)

# Ajouter la colonne "Position (m)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['Position (m)'] = 0

# Calculer la colonne "Position (m)" en utilisant la formule spécifiée
    for i in range(1, len(new_dataframe)):
        new_dataframe.loc[i, 'Position (m)'] = new_dataframe.loc[i, 'vitesse_model'] * (new_dataframe.loc[i, 'Seconds'] - new_dataframe.loc[i - 1, 'Seconds']) + new_dataframe.loc[i - 1, 'Position (m)']

    # Ajouter la colonne "Accélération (m/s^2)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['Accélération (m/s²)'] = 0

    # Calculer la colonne "Accélération (m/s^2)" en utilisant la dérivée du modèle spécifiée
    new_dataframe['Accélération (m/s²)'] = (params_optimises[0] / params_optimises[1]) * np.exp(-(new_dataframe['Seconds'] - params_optimises[2]) / params_optimises[1])

# Ajouter la colonne "F Hzt (N)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['F Hzt (N)'] = 0

# Calculer la colonne "F Hzt (N)" en utilisant la formule spécifiée
    new_dataframe['F Hzt (N)'] = masse * new_dataframe['Accélération (m/s²)']

# Ajouter la colonne "F air (N)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['F air (N)'] = 0

# Calculer la colonne "F air (N)" en utilisant la formule spécifiée
    new_dataframe['F air (N)'] = k * new_dataframe['vitesse_model'] * new_dataframe['vitesse_model'] 

# Ajouter la colonne "F Hzt total (N)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['F Hzt total (N)'] = 0

# Calculer la colonne "F Hzt total (N)" en utilisant la formule spécifiée
    new_dataframe['F Hzt total (N)'] = new_dataframe['F Hzt (N)'] + new_dataframe['F air (N)']

# Ajouter la colonne "F Hzt total (N/kg)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['F Hzt total (N/kg)'] = 0

# Calculer la colonne "F Hzt total (N/kg)" en utilisant la formule spécifiée
    new_dataframe['F Hzt total (N/kg)'] = new_dataframe['F Hzt total (N)'] / masse

# Ajouter la colonne "Puissance (W/kg)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['Puissance (W/kg)'] = 0

# Calculer la colonne "Puissance (W/kg)" en utilisant la formule spécifiée
    new_dataframe['Puissance (W/kg)'] = new_dataframe['F Hzt total (N/kg)'] * new_dataframe['vitesse_model'] 

# Ajouter la colonne "F résultante (N)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['F résultante (N)'] = 0

# Calculer la colonne "F résultante (N)" en utilisant la formule spécifiée
    new_dataframe['F résultante (N)'] = np.sqrt(np.square(new_dataframe['F Hzt total (N)']) + np.square(masse * 9.81))

# Ajouter la colonne "RF (%)" au nouveau DataFrame avec des valeurs initiales nulles
    new_dataframe['RF (%)'] = 0

# Calculer la colonne "RF (%)" en utilisant la formule spécifiée
    new_dataframe['RF (%)'] = new_dataframe['F Hzt total (N)']/new_dataframe['F résultante (N)']*100

# Appliquer la condition : si la colonne des temps est inférieure à 0.3, alors RF (%) = 0
    new_dataframe['RF (%)'] = np.where(new_dataframe['Seconds'] < 0.3, 0, new_dataframe['RF (%)'])

# Arrondir toutes les valeurs du dataframe à deux décimales
    new_dataframe = new_dataframe.round(2)

    # Afficher les paramètres optimisés avec deux chiffres après la virgule
    #st.write("Paramètres optimisés:")
    st.write('<span style="color: green;">Faites défiler vers le bas pour accéder aux différents résultats.</span>', unsafe_allow_html=True)
    st.write(f"- Vmax (vitesse maximale) : {round(params_optimises[0], 2)} m/s")
    st.write(f"- Tau (constante de temps) : {round(params_optimises[1], 2)} s")
    st.write(f"- Delay (délai) : {round(params_optimises[2], 2)} s")

    # Afficher la somme de square difference pour le nouveau DataFrame
    st.write(f"- Somme des carrés des différences : {round(somme_square_difference)}")

  # Tracer le graphique pour le nouveau DataFrame
    plt.plot(new_dataframe['Seconds'], new_dataframe['Vitesse_Reelle'], label='Vitesse brute',color="blue")
    plt.plot(new_dataframe['Seconds'], new_dataframe['vitesse_model'], label='Vitesse modélisée',color="green")
    plt.xlabel('Temps (s)')
    plt.ylabel('Vitesse (m/s)')
    plt.legend()

    # Définir les limites de l'axe des x entre 0 et 8
    plt.xlim(0, new_dataframe['Seconds'].max()+1)

    st.pyplot(plt)

# Afficher le tableau de données pour le nouveau DataFrame
#st.write("Tableau de données :")
#st.dataframe(new_dataframe)

# Calculer la moyenne des colonnes 'HDOP' et '#Sats'
moyenne_hdop = new_dataframe['HDOP'].mean()
moyenne_sats = new_dataframe[' #Sats'].mean()

# Afficher les moyennes
st.write('<span style="color: green;">Qualité du signal :</span>', unsafe_allow_html=True)
st.write(f"En moyenne la dispersion des satellites (HDOP) vaut : {round(moyenne_hdop, 2)}")
st.write(f"Le nombre moyen de satellites est de : {round(moyenne_sats, 2)}")

st.write('<span style="color: green;">Principaux résultats :</span>', unsafe_allow_html=True)

# Calcul de la pente (Drf) en utilisant la régression linéaire
pente_Drf = np.polyfit(new_dataframe['RF (%)'], new_dataframe['vitesse_model'], 1)[0]
st.write(f"- Drf : {round(pente_Drf,2)}")

# Drf en %
Drf = pente_Drf*100
st.write (f"- Drf (%) : {round(Drf,2)}")

# Calcul de F0(N) en utilisant la régression linéaire
ordonnee_origine = np.polyfit(new_dataframe['vitesse_model'],new_dataframe['F Hzt total (N)'], 1)[1]
st.write(f"- F0 (N) : {round(ordonnee_origine,2)}")

# Calcul de F0 (N/kg)
F0 = ordonnee_origine/masse
st.write(f"- F0 (N/kg) : {round(F0,2)}")

# Calcul pente_profil 
pente_profil = np.polyfit(new_dataframe['vitesse_model'],new_dataframe['F Hzt total (N/kg)'],1)[0]
st.write(f"- Pente du profil : {round(pente_profil,2)}")

# Calcul de V0(m/s)
V0 = -F0/pente_profil
st.write(f"- V0 (m/s) : {round(V0,2)}")

# Calcul de Pmax (W)
Pmax = ordonnee_origine*V0/4
st.write(f"- Pmax (W) : {round(Pmax,2)}")

# Calcul Pmax(W/kg)
Pmax_kg = Pmax/masse
st.write(f"- Pmax (W/kg) : {round(Pmax_kg,2)}")

# Calcul de RFmax
Rfmax = new_dataframe['RF (%)'].max()
st.write(f"- RFmax : {round(Rfmax,2)}")


# Supposons que vous souhaitez trouver les temps pour les positions spécifiques
positions_recherchees = [5, 10, 20, 30]

# Initialiser une liste pour stocker les temps correspondants
temps_pour_positions = []

# Parcourir les positions recherchées
for position in positions_recherchees:
    # Trouver l'indice de la première occurrence où la position dépasse ou est égale à la position recherchée
    indice_position = (new_dataframe['Position (m)'] >= position).idxmax()
    
    # Récupérer le temps correspondant à cette position
    temps_correspondant = new_dataframe.loc[indice_position, 'Seconds']
    
    # Ajouter le temps à la liste
    temps_pour_positions.append((position, temps_correspondant))

for position, temps in temps_pour_positions:
      temps_arrondi = round(temps, 2)
      st.write(f"- Temps au {position} m : {temps_arrondi} s")

st.write('<span style="color: green;">Copiez la ligne du tableau ci-dessous et collez la dans le fichier Excel récapitulatif.</span>', unsafe_allow_html=True)

# Créer un dictionnaire pour stocker les résultats importants
results_dict = {
    "Nom Prénom":nom,
    "Num Sprint":num_sprint,
    "Date du test":date_du_test,
    "Date de naissance":date_de_naissance,
    "Age":"",
    "Equipe":equipe,
    "Poste":poste,
    "Statut":statut,
    "Poids":masse,
    "Taille":taille,
    "Température":temperature,
    "Pression":Pb,
    "Vmax (m/s)": round(params_optimises[0], 2),
    "F0 (N/kg)": round(F0, 2),
    "V0 (m/s)": round(V0, 2),
    "Pmax (W/kg)": round(Pmax_kg, 2),
    "Pente du profil": round(pente_profil, 2),
    "Drf": round(Drf, 2),
}

# Ajouter les temps pour chaque position au dictionnaire
for position, temps in temps_pour_positions:
    results_dict[f"T {position}m"] = round(temps, 2)

# Afficher le tableau avec les résultats
st.table([results_dict])


# Créer les points pour la force
x_force_points = [V0, 0]
y_force_points = [0, F0]

# Ajuster un polynôme de degré 1 (polynôme linéaire) aux points de force
coefficients_force = np.polyfit(x_force_points, y_force_points, 1)

# Créer la fonction polynomiale pour la force
poly_function_force = np.poly1d(coefficients_force)

# Générer des valeurs intermédiaires pour une courbe lisse de force
x_force_smooth = np.linspace(0, V0, 100)
y_force_smooth = poly_function_force(x_force_smooth)

# Créer les points pour la puissance
x_power_points = [0, V0, V0/2]
y_power_points = [0, 0, Pmax_kg]

# Ajuster un polynôme de degré 2 (polynôme quadratique) aux points de puissance
coefficients_power = np.polyfit(x_power_points, y_power_points, 2)

# Créer la fonction polynomiale pour la puissance
poly_function_power = np.poly1d(coefficients_power)

# Générer des valeurs intermédiaires pour une courbe lisse de puissance
x_power_smooth = np.linspace(0, V0, 100)
y_power_smooth = poly_function_power(x_power_smooth)

# Tracer le graphique
fig, ax1 = plt.subplots()
ax1.set_xlabel('Vitesse (m/s)')
ax1.set_ylabel('Force (N/kg)', color='purple')
ax1.scatter([V0], [0], color='purple')  # Point V0
ax1.scatter([0], [F0], color='purple')  # Point F0
ax1.plot(x_force_smooth, y_force_smooth, color='purple', linestyle='--', label='Force')  # Ligne de force

# Ajouter la puissance 
ax2 = ax1.twinx()
ax2.set_ylabel('Puissance (W/kg)', color='orange')
ax2.scatter([0], [0], color='orange')
ax2.scatter([V0], [0], color='orange')
ax2.scatter([V0/2], [Pmax_kg], color='orange')
ax2.plot(x_power_smooth, y_power_smooth, color='orange', linestyle='--', label='Puissance')  # Courbe de puissance


# Ajouter des étiquettes et une légende
ax1.set_title('Profil F-V')

# Afficher le graphique dans Streamlit
st.pyplot(fig)

# Données brutes
st.write('<span style="color: green;">Voici le tableau contenant les données brutes de cette analyse.</span>', unsafe_allow_html=True)
st.dataframe(new_dataframe)
