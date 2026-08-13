# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

* **Institution :** [Institut Supérieur Polytechnique de Madagascar][www.ispm-edu.com](http://www.ispm-edu.com)

* **Thème du projet :** Prédiction de l’annulation d’une réservation hôtelière
  
## **I. Informations sur le Groupe**
* **Nom du groupe de projet :** `NEXUS`

### Membres de l'équipe

| Numéro | Nom Complet | Classe | Rôle |
| :--- | :--- | :--- | :--- |
| 06 | RAZAFIMANANTSOA Nathalie Malalasoa Kantoniaina| ISAIA 4 | Data Scientist |
| 07 | ANDRIAMANARIVO Tahiana Miora | ISAIA 4 | Rédacteur & Pitcher (README & Vidéo) |
| 08 | RATSIMISETRA Hasiniaina | ISAIA 4 | Data scientist et Présentateur |
| 11 | RAZANADRAKOTO Noël Patrick | ISAIA 4  | Feature Engineer |
| 12 | RAMIARANJAKAHARIMANANA Mendrika Harinjato | ISAIA 4 | Rédacteur & Pitcher (README & Vidéo) |
| 13 | RATSIMAHOLINANDRASANA Antsa Nifaliana| ISAIA 4 | Responsable de la modélisation |

---

## **II. Résumé du Travail**

### Problématique

Atlantic Haven Hotels fait face à un taux d’annulation élevé, ce qui rend difficile de prévoir le taux d’occupation réel de ses établissements. Cela entraîne des pertes de revenus, notamment lorsque les chambres annulées trop tard ne peuvent plus être relouées. Dans ce contexte, prédire à l’avance la variable `reservation_annulee` est très utile. En identifiant les réservations susceptibles d’être annulées, le groupe peut remettre les chambres en vente plus tôt, mieux gérer le surbooking et améliorer sa stratégie de *yield management*. L’objectif est d’augmenter le taux d’occupation et les revenus, tout en évitant les excès de surbooking et en maintenant une bonne expérience client.

### Méthodologie adoptée

#### &nbsp;&nbsp;&nbsp;&nbsp; 1. Analyse Exploratoire des Données (EDA)
L'analyse exploratoire s'est concentrée sur deux axes majeurs :
* **Analyse de la variable cible (`reservation_annulee`)** : Vérification du type (`int64`), contrôle des valeurs manquantes (aucune valeur nulle détectée) et étude de la distribution des classes (annulée vs non annulée) afin d'anticiper d'éventuels déséquilibres.
* **Analyse des variables explicatives** : 
  * Inspection des distributions des variables numériques à l'aide d'histogrammes et de *boxplots* pour identifier la forme des distributions et détecter les valeurs atypiques (*outliers*).
  * Analyse des fréquences des variables catégorielles via des diagrammes en barres (*countplots*).
  * Étude de l'impact temporel et de la saisonnalité (analyse du taux d'annulation moyen en fonction du jour de la semaine et du mois de réservation ou d'arrivée).

---

#### &nbsp;&nbsp;&nbsp;&nbsp; 2. Traitement des Données et Nettoyage
Les traitements d'imputation appliqués pour pallier les valeurs manquantes s'articulent comme suit :
* **Variables catégorielles / Identifiants** : Remplacement des valeurs manquantes dans `agent_id` par la catégorie `'Non_spécifié'`.
* **Variables numériques discrètes** : Imputation par `0` pour `enfants` et `demandes_speciales`.
* **Variables numériques continues** : Imputation de `prix_moyen_nuit_eur` par sa **médiane** afin d'éviter le biais lié aux valeurs extrêmes.
* **Variables catégorielles de marché** : Imputation de `marche_origine` par son **mode**.

---

#### &nbsp;&nbsp;&nbsp;&nbsp; 3. Ingénierie des Caractéristiques (*Feature Engineering*)
Afin d'enrichir le signal prédictif transmis aux modèles, plusieurs catégories de caractéristiques ont été créées :
* **Caractéristiques temporelles et de durée** :
  * Extraction du jour de la semaine, du mois et de l'année à partir des dates `date_reservation` et `date_arrivee`.
  * Calcul de la durée totale du séjour (`nuits_sejour = date_arrivee - date_reservation`).
* **Ratios et indicators comportementaux** :
  * Création d'un indicateur binaire `client_recurrent` basant son état sur l'historique de réservation (`reservations_passees` et `annulations_passees`).
  * Calcul du ratio d'annulation passée (`ratio_annulation_passee`) et du ratio enfants/adultes (`ratio_enfants_adultes`).
* **Encodage et Normalisation** :
  * Encodage binaire des variables catégorielles via **One-Hot Encoding** (`pd.get_dummies`).
  * Normalisation des variables numériques continues à l'aide de `StandardScaler`.

---

#### &nbsp;&nbsp;&nbsp;&nbsp; 4. Validation Temporelle (*Temporal Split*)
Pour éviter la fuite d'information (*data leakage*) et refléter des conditions réelles de déploiement où l'on prédit le futur à partir du passé :
1. Les données d'entraînement ont été triées chronologiquement selon la colonne `date_reservation`.
2. Un découpage strict a été appliqué : **80 %** des données les plus anciennes dédiées à l'entraînement (`X_train_raw`, `y_train_raw`) et **20 %** des données les plus récentes réservées à la validation (`X_val_raw`, `y_val_raw`).
3. La mise à l'échelle (`StandardScaler`) a été entraînée (*fitted*) **uniquement** sur l'ensemble d'entraînement avant d'être appliquée (*transformed*) sur la validation et le test.

---

#### &nbsp;&nbsp;&nbsp;&nbsp; 5. Modèle Baseline et Modèles Comparés
Plusieurs architectures de modèles ont été expérimentées et comparées :
* **Baseline – Régression Logistique** : Utilisée comme modèle de référence interprétable et optimisée avec `HalvingRandomSearchCV` puis via `Optuna`.
* **Algorithmes d'Ensemble par Arbres** :
  * **Random Forest Classifier** : Pour capturer les relations non linéaires.
  * **XGBoost Classifier** : Ajusté en tenant compte du ratio de classes via le paramètre `scale_pos_weight`.
  * **LightGBM Classifier** : Entraîné et réglé finement avec `Optuna` pour maximiser la performance prédictive globale.

---

#### &nbsp;&nbsp;&nbsp;&nbsp; 6. Choix du Seuil de Décision et Métrique d'Optimisation
* **Métrique Cible** : La métrique priorisée lors des recherches d'hyperparamètres (notamment via `Optuna`) est le **F1-Score**, garantissant un équilibre optimal entre la précision (éviter les fausses alarmes) et le rappel (détecter un maximum d'annulations réelles).
* **Gestion du Déséquilibre et Seuil** : L'ajustement du poids des classes (`class_weight='balanced'` ou calcul dynamique du `scale_pos_weight`) ainsi que l'optimisation directe de la probabilité prédite par rapport au rappel permettent d'ajuster le seuil de classification décisionnel pour maximiser la métrique globale sur le jeu de validation temporel.

### Résultats obtenus

*(Indiquez le meilleur F1-score obtenu sur votre jeu de validation, les principales métriques complémentaires et une découverte importante issue de votre analyse.)*

### Mots-clés

Regression logistique, annulation, validation temporelle, F1-score, feature engineering

---

## **III. Contenu du Repository**

Voici la liste des fichiers et liens importants permettant d’évaluer votre travail :

- **notebook.ipynb** : code complet de l’EDA, du prétraitement, de la modélisation et de l’évaluation ;
- **submission.csv** : prédictions sur `reservations_test.csv` ;
- **README.md** : présent rapport complété ;
- **requirements.txt** : dépendances nécessaires à la reproduction du projet *(si nécessaire)* ;
- *(ajoutez ici les autres fichiers utiles sans inclure les fichiers temporaires).* 

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION** — Google Drive ou YouTube](https://www.youtube.com/)
- [Lien vers le dépôt GitHub](https://github.com/)
- [Lien vers une autre ressource — facultatif](https://www.google.com/)

---

## **IV. Résultats de Modélisation**

Présentez les résultats obtenus sur **le même jeu de validation** afin que la comparaison soit valide.

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline |  |  |  |  |  |
| Modèle 2 |  |  |  |  |  |
| Modèle 3 |  |  |  |  |  |
| Modèle final |  |  |  |  |  |

**Seuil de décision retenu :** *(votre réponse ici)*

**Justification du choix du modèle final :**

*(Votre réponse ici. Ne vous limitez pas au score : considérez la stabilité, l’interprétabilité, les erreurs et le coût métier.)*

---

## **V. Réponses aux Questions d’Analyse**

*Répondez précisément aux questions ci-dessous. Utilisez des chiffres, tableaux ou références à vos graphiques pour justifier vos réponses.*

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l’accuracy pour cette tâche ?**

L’accuracy peut très vite nous induire en erreur. Dans le secteur hôtelier, les réservations maintenues sont généralement bien plus nombreuses que les annulations. Un modèle paresseux qui prédirait que personne n'annule jamais afficherait un taux de bonnes réponses très élevé, tout en étant totalement incapable de repérer les vrais annulations.

C'est là que le F1-score prend tout son sens. En combinant la précision (quand le modèle prédit une annulation, a-t-il raison ?) et le rappel (parmi toutes les vraies annulations, combien en a-t-on attrapées ?), il cherche un juste équilibre. Pour l'hôtel, l'enjeu est clair : capturer un maximum d'annulations sans déclencher une pluie de fausses alertes.


#### **Q2. Dans ce contexte, qu’est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

Les deux erreurs ne se valent pas et dépendent avant tout de la stratégie de l'hôtel.

* **Le Faux Négatif** (la non-détection) : C'est le client qui annule au dernier moment alors que le modèle pensait qu'il viendrait. Financièrement, c'est un coup dur : la chambre reste vide, bloquée trop tard pour être relouée.

* **Le Faux Positif** (la fausse alerte) : C'est prédire une annulation alors que le client finit par se présenter.

Si l'hôtel utilise notre modèle pour envoyer de simples rappels de politesse, un faux positif n'a aucun impact négatif. En revanche, si l'hôtel pratique du surbooking agressif et qu'un client fidèle arrive sans trouver de chambre disponible, les dégâts sur sa réputation seront bien plus graves que le coût d'une chambre vide.


#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

*(Listez les variables, expliquez leur construction et quantifiez le gain observé.)*

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Utiliser un simple train_test_split aléatoire aurait faussé nos résultats par du data leakage (fuite de données). En mélangeant le passé et le futur, le modèle apprendrait à partir d'événements futurs pour "prédire" des événements passés, affichant des performances anormalement élevées en laboratoire, mais décevantes sur le terrain.

Nous avons donc opté pour une validation temporelle stricte :

> ***Entraînement*** : Les 80 % de réservations les plus anciennes chronologiquement.

> ***Validation*** : Les 20 % les plus récentes.

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

- *(profil ou scénario 1)*
- *(profil ou scénario 2)*
- *(profil ou scénario 3)*
- *(...)*

*Attention : décrivez des circonstances observables et des interactions entre variables. Ne présentez pas une région ou une population comme étant intrinsèquement à risque.*

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l’entraînement ?**

*(Votre réponse ici. Précisez comment vous avez évité la fuite de données.)*

#### **Q7. Selon vous, quelle action l’hôtel devrait-il entreprendre lorsqu’une réservation en cours présente une forte probabilité d’annulation ?**

*(Votre réponse ici. Proposez une intervention proportionnée qui n’annule pas automatiquement la réservation du client.)*

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

*(Présentez au moins une comparaison chiffrée et discutez les limites liées aux petits sous-groupes.)*

#### **Q9. Analyse des erreurs**

Analysez au minimum :

- cinq faux positifs ;
- cinq faux négatifs ;
- les raisons possibles de ces erreurs ;
- une piste d’amélioration des données ou du modèle.

*(Votre réponse ici.)*

---

## **VI. Conclusion et Recommandations**

*(Résumez en un court paragraphe les performances, les limites et les conditions raisonnables d’utilisation du modèle.)*

**Recommandation opérationnelle finale :**

*(Votre réponse ici.)*

---

## **VII. Reproductibilité**

- version de Python : Python 3.11.9
- principales bibliothèques (et versions) : sklearn, numpy, matplotlib, pandas, seaborn
- graine(s) aléatoire(s) :
- commande ou procédure d’exécution :
- durée approximative d’entraînement :
- environnement utilisé : Google Colab

---

## **VIII. Bibliographie**

### &nbsp;&nbsp;&nbsp;&nbsp; 1. Documentations et liens
- *`Sklearn documentation`* — https://scikit-learn.org/0.21/documentation.html
- *`LightGBM`* — https://lightgbm.readthedocs.io/en/stable/Parameters.html
- *`XGBoost`* — https://xgboost.readthedocs.io/en/stable/
- *`Matplotlib`* — https://matplotlib.org/stable/index.html

### &nbsp;&nbsp;&nbsp;&nbsp; 2. Assistants IA

#### Gemini
- **Cadrage & Rédaction :** Structuration de la problématique métier, de l'analyse d'erreurs et rédaction du rapport.
- **Orientation méthodologique :** Prise en main du protocole de validation temporelle et stratégie d'optimisation du F1-score.

#### Claude Code
- **Développement & Refactoring :** Écriture des scripts Python, création des pipelines et modularisation du code.
- **Qualité & Conformité :** Vérification automatisée de la reproductibilité.
--
### &nbsp;&nbsp;&nbsp;&nbsp; 3. License
* **Code source :** Distribué sous licence **MIT**
* **Données :** Les données fournies (`reservations_train.csv`, `reservations_test.csv`) sont entièrement synthétiques, fournies par l'ISPM à des fins strictement pédagogiques pour l'examen final S2 (Année universitaire 2025-2026).
* **Usage :** Ce dépôt est destiné à l'évaluation académique du cours de Machine Learning & Data Science (Master 1 - ISPM).

