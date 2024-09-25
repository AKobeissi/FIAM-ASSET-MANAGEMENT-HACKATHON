import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Lire le fichier CSV avec spécification des types de données
df = pd.read_csv('/content/hackathon_sample_v2.csv', sep=';')
# Gérer les valeurs manquantes en les remplissant avec la moyenne de la colonne
df=df.dropna()

# Définir la variable dépendante (excess return) et les variables explicatives
y = df['stock_exret']
X = df[[ 'eqnetis_at', 'div12m_me','rf', 'ret_1_0','rd_sale','saleq_su', 'ret_12_1', 'sale_gr1']] #Je les ai selectioné arbitrairement et enlever celle non significative (mais mauvais pratique de faire comme ça)

# Ajouter une constante pour le modèle de régression
X = sm.add_constant(X)

# Ajuster le modèle de régression linéaire
model = sm.OLS(y, X).fit()

# Obtenir les résultats avec des erreurs standard robustes
robust_results = model.get_robustcov_results(cov_type='HC1')

# Résumé des résultats robustes
print(robust_results.summary())

# Pour checker la multicolinéarité entre nos variables
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] 
print(vif_data)
