#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[43]:


df=pd.read_csv("transactions_TP2.csv")
print(df.shape)
print(df.head())
print(df.dtypes)


# In[44]:


print(df.isna().sum())


# In[45]:


print(df.isna().mean()*100)


# In[46]:


manq=pd.DataFrame({
    "nombre":df.isna().sum(),
    "pourcentage":(df.isna().mean()*100).round(2)
})
manq=manq.sort_values("pourcentage",ascending=False)
print(manq)


# In[47]:


nb_doublons=df.duplicated(subset="id_operation").sum()
print("nombre de doublons:",nb_doublons)


# In[48]:


df=df.drop_duplicates(subset="id_operation")
print("lignes restantes:",df.shape[0])


# In[49]:


plt.figure(figsize=(8,4))
plt.hist(df["montant_dh"].dropna(),bins=50)
plt.title("Histogramme de montant_dh")
plt.xlabel("montant_dh")
plt.ylabel("Nombre d'operations")
plt.show()


# In[50]:


plt.figure(figsize=(8,3))
sns.boxplot(x=df["montant_dh"])
plt.title("Boxplot de montant_dh")
plt.show()


# In[51]:


print(df["montant_dh"].describe())
print(df["montant_dh"].nlargest(5))


# In[52]:


print(df["fraude"].value_counts())
print(df["fraude"].value_counts(normalize=True)*100)
plt.figure(figsize=(5,3))
sns.countplot(x="fraude",data=df)
plt.title("Repartition des classes")
plt.show()


# In[53]:


print(pd.crosstab(df["resultat_enquete"],df["fraude"]))


# In[54]:


X=df.drop(columns=["id_operation","id_client","resultat_enquete","fraude"])
y=df["fraude"]
print(X.shape)
print(X.columns.tolist())


# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.20, stratify=y, random_state=42)

print("Reference:",X_train.shape)
print("Verification:",X_test.shape)

print(y_train.value_counts(normalize=True)*100)
print(y_test.value_counts(normalize=True)*100)


# In[56]:


num_cols= X_train.select_dtypes(include="number").columns.tolist()
cat_cols=X_train.select_dtypes(exclude="number").columns.tolist()
print("Numerique:", num_cols)
print("Categorielles:", cat_cols)


# In[57]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
X_train_num = num_pipe.fit_transform(X_train[num_cols])
X_test_num=num_pipe.transform(X_test[num_cols])


# In[58]:


print("Medianes :", num_pipe.named_steps["imputer"].statistics_)
print("Moyennes:", num_pipe.named_steps["scaler"].mean_)
print("Ecarts-types:", num_pipe.named_steps["scaler"].scale_)


# In[59]:


print(np.isnan(X_train_num).sum(), np.isnan(X_test_num).sum())


# In[60]:


from sklearn.preprocessing import OneHotEncoder

cat_pipe= Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("OneHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


# In[61]:


from sklearn.compose import ColumnTransformer

prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

X_train_prep = prep.fit_transform(X_train)
X_test_prep = prep.transform(X_test)

print("Référence avant :", X_train.shape, "→ après :", X_train_prep.shape)
print("Vérification avant :", X_test.shape, "→ après :", X_test_prep.shape)


# In[62]:


print(prep.get_feature_names_out())
print(np.isnan(X_train_prep).sum(), np.isnan(X_test_prep).sum())


# In[63]:


from sklearn.preprocessing import RobustScaler

num_pipe_rob = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])
num_pipe_rob.fit(X_train[num_cols])

exemples = pd.DataFrame({
    "montant_dh": [700, 30000],
    "anciennete_mois": [87, 87],
    "nb_ops_24h": [3, 3],
    "heure": [11, 11]
})[num_cols]

print("StandardScaler :", num_pipe.transform(exemples)[:, 0])
print("RobustScaler   :", num_pipe_rob.transform(exemples)[:, 0])


# In[64]:


Q1= X_train["montant_dh"].quantile(0.25)
Q3=X_train["montant_dh"].quantile(0.75)
IQR=Q3-Q1
print("Q1=", Q1, "|Q3=", Q3, "|IQR=", IQR)


# In[65]:


L= Q1-1.5 * IQR
U= Q3+1.5 * IQR
out= X_train[(X_train["montant_dh"]<L)|(X_train["montant_dh"]>U)]

print("limites :",L,U)
print("Lignes concernees :", len(out))
print(out["montant_dh"].sort_values(ascending=False).head(10))


# In[66]:


montant_clip = X_train["montant_dh"].clip(lower=L, upper=U)

fig, axes = plt.subplots(1,2, figsize=(12,4))
sns.boxplot(x=X_train["montant_dh"], ax=axes[0])
axes[0].set_title("montants conserves")
sns.boxplot(x=montant_clip, ax=axes[1])
axes[1].set_title("Montants limites(L et U)")
plt.tight_layout()
plt.show()


# In[67]:


from sklearn.preprocessing import MinMaxScaler , StandardScaler, RobustScaler

col=["montant_dh"]
mm= MinMaxScaler().fit(X_train[col])
sd= StandardScaler().fit(X_train[col])
rb= RobustScaler().fit(X_train[col])

mm_ref= mm.transform(X_train[col])[:, 0]
sd_ref= sd.transform(X_train[col])[:, 0]
rb_ref= rb.transform(X_train[col])[:, 0]

mm_ver= mm.transform(X_test[col])[:, 0]
sd_ver= sd.transform(X_test[col])[:, 0]
rb_ver= rb.transform(X_test[col])[:, 0]


# In[68]:


exemples= pd.DataFrame({"montant_dh":[300.0, 700.0, 1200.0, 30000.0]})

tableau= pd.DataFrame({
    "Montant(DH)": exemples["montant_dh"],
    "Min-Max": mm.transform(exemples)[:, 0],
    "Z-score": sd.transform(exemples)[:, 0],
    "Robust": rb.transform(exemples)[:, 0]
}).round(3)
print(tableau)


# In[69]:


fig, axes= plt.subplots(1, 3, figsize=(15, 4))
donnees= [("Min-Max", mm_ref), ("Z-score", sd_ref), ("RobustScaler", rb_ref)]

for ax, (nom, vals) in zip(axes, donnees):
    ax.hist(pd.Series(vals).dropna(), bins=50)
    ax.set_title(nom)
    ax.set_xlabel("montant transforme")
plt.tight_layout()
plt.show()


# In[70]:


ex_train= pd.DataFrame({"x":[100.0, 200.0, 300.0]})
s= MinMaxScaler().fit(ex_train)
print(s.transform(pd.DataFrame({"x": [400.0]})))

print(mm.transform(pd.DataFrame({"montant_dh":[40000.0]})))


# In[71]:


compte= y_train.value_counts()
print(compte)
print("Rapport:", compte[0] / compte[1])


# In[72]:


from imblearn.over_sampling import RandomOverSampler

ros= RandomOverSampler(random_state=42)
X_over, y_over= ros.fit_resample(X_train_prep, y_train)

print("Avant:", y_train.value_counts().to_dict())
print("Apres:", pd.Series(y_over).value_counts().to_dict())
print("Lignes dupliquees:", pd.DataFrame(X_over).duplicated().sum())


# In[73]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train_prep, y_train)

print("Avant :", len(y_train), "lignes")
print("Apres :", len(y_under), "lignes", pd.Series(y_under).value_counts().to_dict())
print("Lignes supprimees :", len(y_train) - len(y_under))


# In[74]:


from imblearn.over_sampling import SMOTE

X_toy = np.array([[100, 1], [120, 3], [110, 2], [130, 4],      
                  [20, 1], [25, 1], [30, 2], [22, 1], [28, 2],
                  [35, 1], [40, 2], [45, 3], [50, 2], [55, 1]]) 
y_toy = np.array([1]*4 + [0]*10)

sm = SMOTE(k_neighbors=2, random_state=42)
X_res, y_res = sm.fit_resample(X_toy, y_toy)

print("Avant :", len(X_toy), "| Apres :", len(X_res))
print("Fraudes reelles :\n", X_toy[:4])
print("Fraudes synthetiques :\n", X_res[len(X_toy):])


# In[75]:


tableau = pd.DataFrame({
    "Avant": y_train.value_counts(),
    "Apres surechantillonnage": pd.Series(y_over).value_counts(),
    "Apres sous-echantillonnage": pd.Series(y_under).value_counts()
})
print(tableau)
print("Verification (inchange) :", y_test.value_counts().to_dict())


# In[76]:


from joblib import dump, load
from sklearn.preprocessing import RobustScaler

num_final = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])
cat_final = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

prep_final = ColumnTransformer([
    ("num", num_final, num_cols),
    ("cat", cat_final, cat_cols)
])


# In[77]:


X_train_final = prep_final.fit_transform(X_train)
X_test_final = prep_final.transform(X_test)

print(X_train_final.shape, X_test_final.shape)
print("Valeurs manquantes :", np.isnan(X_train_final).sum(), np.isnan(X_test_final).sum())


# In[78]:


ligne = X_test.iloc[[0]].copy()
ligne["canal"] = "paiement_mobile"      

resultat = prep_final.transform(ligne)
noms = prep_final.get_feature_names_out()
print([(n, v) for n, v in zip(noms, resultat[0]) if "canal" in n])


# In[79]:


dump(prep_final, "transformateur_final.joblib")
prep_charge = load("transformateur_final.joblib")

echantillon = X_test.head(5)
print(np.allclose(prep_final.transform(echantillon), prep_charge.transform(echantillon)))


# In[80]:


print("Doublons retires :", 45)
print("Valeurs manquantes (apres doublons) :", df.isna().sum().sum())


# In[81]:


col = X_train["montant_dh"]

print("Moyenne :", col.mean(), "| Mediane :", col.median())

col_ext = pd.concat([col, pd.Series([1_000_000.0])])

print("Avec un montant extreme -> Moyenne :", col_ext.mean(), "| Mediane :", col_ext.median())


# In[82]:


from imblearn.over_sampling import SMOTENC

ex = X_train[["montant_dh", "nb_ops_24h", "canal", "appareil"]].join(y_train).dropna()
X_ex = ex[["montant_dh", "nb_ops_24h", "canal", "appareil"]].astype({"canal": "object", "appareil": "object"})
y_ex = ex["fraude"]

smotenc = SMOTENC(categorical_features=[2, 3], random_state=42)
X_res, y_res = smotenc.fit_resample(X_ex, y_ex)

print("Avant :", y_ex.value_counts().to_dict())
print("Après :", pd.Series(y_res).value_counts().to_dict())
print("Canaux avant :", sorted(X_ex["canal"].unique()))
print("Canaux après :", sorted(X_res["canal"].unique()))
print("Appareils après :", sorted(X_res["appareil"].unique()))

