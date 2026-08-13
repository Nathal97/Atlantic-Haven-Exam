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

* **Meilleur F1-Score (Validation) :** **`0.4766`** (obtenu avec la **Régression Logistique** sur les données enrichies par *Feature Engineering*).
* **Principales métriques complémentaires :**
  * **Précision :** `0.3725` ($37.25\%$)
  * **Rappel :** `0.6613` ($66.13\%$)
  * **ROC-AUC :** `0.6742`
  * **Matrice de confusion :** `689` Vrais Négatifs, `285` Vrais Positifs, `480` Faux Positifs et `146` Faux Négatifs sur un total de 1 600 réservations de validation.

* **Découverte importante issue de l'analyse :**
  L'application du *Feature Engineering* (création de ratios comportementaux, variables temporelles et statut client) a permis à la **Régression Logistique** d'améliorer significativement sa Précision (passant de $0.3538$ à $0.3725$) et sa ROC-AUC (de $0.6623$ à $0.6742$), lui permettant de surpasser les modèles d'ensemble plus complexes (LightGBM et Random Forest) en termes d'équilibre global ($F1$-score). De plus, l'ajustement du seuil de décision autour de `0.35`–`0.40` s'est révélé indispensable pour capturer efficacement plus de $66\%$ des annulations réelles face au déséquilibre des classes.

### Mots-clés

Regression logistique, LightGBM, Random Forest, validation temporelle, F1-score, feature engineering, 

---

## **III. Contenu du Repository**

Voici la liste des fichiers et liens importants permettant d’évaluer votre travail :

- **EXAM.ipynb** : code complet de l’EDA, du prétraitement, de la modélisation et de l’évaluation ;
- **submission.csv** : prédictions sur `reservations_test.csv` ;
- **README.md** : le rapport ;
- **requirements.txt** : dépendances nécessaires à la reproduction du projet *(si nécessaire)* ;

**🔗 Liens utiles :**

- Lien vers la vidéo de présentation — Google Drive (https://drive.google.com/drive/folders/11k-bcwkn8nblvOZ3iNHbIZaRQGAUpQEM)
- Lien vers le dépôt GitHub (https://github.com/Nathal97/Atlantic-Haven-Exam/)

---

## **IV. Résultats de Modélisation**

Les résultats obtenus sur **le même jeu de validation** avec différents modèle.

### Comparaison des Modèles (Données simple)

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| **Régression Logistique (Baseline)** | `C=0.0189`, `penalty='l1'`, `solver='liblinear'`, `class_weight='balanced'` | **0.4699** | 0.3538 | 0.6993 | 0.6623 |
| **Random Forest** | `n_estimators=300`, `max_depth=8`, `class_weight='balanced_subsample'` | **0.4761** | 0.3502 | 0.7436 | 0.6465 |
| **LightGBM ** | `n_estimators=500`, `lr=0.0143`, `max_depth=4`, `scale_pos_weight=3.39` | **0.4721** | 0.3578 | 0.7389 | 0.6566 |

### Comparaison des Modèles (application de feature engineering aux données)

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| **Régression Logistique** | `C=0.0567`, `penalty='l1'`, `solver='liblinear'` | **0.4766** | 0.3725 | 0.6613 | 0.6742 |
| **LightGBM** | `n_estimators=400`, `lr=0.0200`, `max_depth=3`, `num_leaves=57` | **0.4762** | 0.3721 | 0.6613 | 0.6592 |
| **Random Forest** | `n_estimators=300`, `max_depth=6`, `class_weight='balanced_subsample'` | **0.4711** | 0.3604 | 0.6798 | 0.6537 |

---

**Seuil de décision retenu :** `0.35` à `0.40` (Ajusté selon la gestion du déséquilibre `/ class_weight` / `scale_pos_weight`)

* **Raisonnement :** Le seuil standard par défaut de `0.50` est sous-optimal face à des données déséquilibrées. En abaissant le seuil décisionnel autour de `0.35`–`0.40`, on privilégie la détection des annulations (augmentation du Rappel) sans dégrader exagérément la Précision, maximisant ainsi le F1-score sur la validation temporelle.

---

**Justification du choix du modèle final :**

Deux approches se distinguent selon les objectifs et priorités métiers de l'établissement :

### 1. Choix Recommandé : **Régression Logistique (avec feature engineering)**
* **Performance globale :** C'est le modèle qui obtient la meilleure performance globale toutes expériences confondues, affichant le **F1-score le plus élevé (0.4766)**, la meilleure **Précision (0.3725)** et la plus forte **ROC-AUC (0.6742)**.
* **Interprétabilité et Stabilité :** Grâce à la pénalité L1 (`penalty='l1'`), ce modèle effectue une sélection automatique des variables les plus pertinentes. Il offre une transparence totale pour les équipes métiers (compréhension claire des facteurs de risque d'annulation) tout en garantissant un coût d'inférence minimal et une très bonne stabilité en production.

### 2. Alternative Axée sur la Détection des Annulations : **Random Forest (Données simple)**
* **Priorité au Rappel :** Si le coût métier d'une annulation non anticipée (chambre vacante non réattribuée) est jugé critique, le **Random Forest sur données simples** est une excellente alternative. Il offre le **Rappel le plus élevé (0.7436)** — détectant plus de 74% des annulations réelles — tout en conservant un F1-score très solide (**0.4761**).

> **Conclusion :** La **Régression Logistique (après feature engineering)** est retenue comme **modèle principal** pour son équilibre parfait entre F1-score, ROC-AUC, précision et explicabilité. Si la priorité absolue est de minimiser au maximum les fausses présences (manquer le moins d'annulations possible), le **Random Forest (données simples)** constitue une alternative métier pertinente.

---

## **V. Réponses aux Questions d’Analyse**

### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l’accuracy pour cette tâche ?**

L’accuracy peut très vite nous induire en erreur. Dans le secteur hôtelier, les réservations maintenues sont généralement bien plus nombreuses que les annulations. Un modèle paresseux qui prédirait que personne n'annule jamais afficherait un taux de bonnes réponses très élevé, tout en étant totalement incapable de repérer les vrais annulations.

C'est là que le F1-score prend tout son sens. En combinant la précision (quand le modèle prédit une annulation, a-t-il raison ?) et le rappel (parmi toutes les vraies annulations, combien en a-t-on attrapées ?), il cherche un juste équilibre. Pour l'hôtel, l'enjeu est clair : capturer un maximum d'annulations sans déclencher une pluie de fausses alertes.


### **Q2. Dans ce contexte, qu’est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

Les deux erreurs ne se valent pas et dépendent avant tout de la stratégie de l'hôtel.

* **Le Faux Négatif** (la non-détection) : C'est le client qui annule au dernier moment alors que le modèle pensait qu'il viendrait. Financièrement, c'est un coup dur : la chambre reste vide, bloquée trop tard pour être relouée.

* **Le Faux Positif** (la fausse alerte) : C'est prédire une annulation alors que le client finit par se présenter.

Si l'hôtel utilise notre modèle pour envoyer de simples rappels de politesse, un faux positif n'a aucun impact négatif. En revanche, si l'hôtel pratique du surbooking agressif et qu'un client fidèle arrive sans trouver de chambre disponible, les dégâts sur sa réputation seront bien plus graves que le coût d'une chambre vide.


### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

### Variables créées
 
**1. ratio_annulations_passees**
Construction : annulations_passees / (reservations_passees + annulations_passees), avec gestion du cas 0/0 (→ 0).
C'est un ratio historique du comportement du client — la variable la plus susceptible d'avoir un pouvoir prédictif fort, car elle résume directement la propension passée à annuler.
 
**2. total_personnes**
Construction : adultes + enfants.
Simple agrégation, utilisée ensuite comme base du ratio suivant.
 
**3. prix_par_personne_par_nuit**
Construction : prix_moyen_nuit_eur / total_personnes (0 si total_personnes = 0).
Combine prix et taille du groupe en une seule variable normalisée.
 
**4. Variables temporelles**
annee_reservation, mois_reservation, jour_semaine_reservation, jour_annee_reservation, et leurs équivalents _arrivee.
Extraites via .dt.year, .dt.month, .dt.dayofweek, .dt.dayofyear sur date_reservation et date_arrivee.
Les graphiques d'analyse montrent une variation visible du taux d'annulation selon le mois et le jour de la semaine, justifiant leur inclusion.
 
### Gain quantifié
 
Comparaison d'une régression logistique *avant* feature engineering (données brutes imputées uniquement) et *après* feature engineering (avec optimisation Optuna) :
 
| Métrique | Avant FE | Après FE | Gain |
|---|---|---|---|
| Accuracy | 0.5769 | 0.6088 | +0.032 |
| Precision | 0.3538 | 0.3725 | +0.019 |
| Recall | 0.6993 | 0.6613 | −0.038 |
| F1-Score | 0.4699 | 0.4766 | +0.007 |
| AUC-ROC | 0.6623 | 0.6742 | +0.012 |
 
Le gain global reste modeste (F1 +1,4 %, AUC-ROC +1,8 %), avec une baisse du recall compensée par une hausse de la precision — cohérent avec l'ajout de variables qui affinent le modèle sans changer radicalement sa capacité de discrimination.

### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Utiliser un simple train_test_split aléatoire aurait faussé nos résultats par du data leakage (fuite de données). En mélangeant le passé et le futur, le modèle apprendrait à partir d'événements futurs pour "prédire" des événements passés, affichant des performances anormalement élevées en laboratoire, mais décevantes sur le terrain.

Nous avons donc opté pour une validation temporelle stricte :

> ***Entraînement*** : Les 80 % de réservations les plus anciennes chronologiquement.

> ***Validation*** : Les 20 % les plus récentes.

### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

#### Scénario 1 : La réservation très anticipée à long délai d'attente (lead_time élevé)

&nbsp;&nbsp;&nbsp;&nbsp; Circonstances : Réservation effectuée plusieurs mois à l'avance sans prépaiement ni demande spéciale.

&nbsp;&nbsp;&nbsp;&nbsp; Interaction : lead_time_jours > 120 jours ET demandes_speciales = 0.

#### Scénario 2 : Le client au comportement d'annulation récurrent

&nbsp;&nbsp;&nbsp;&nbsp; Circonstances : Client ayant un historique d'annulations antérieures non négligeable.

&nbsp;&nbsp;&nbsp;&nbsp; Interaction : annulations_passees > 0 ET reservations_passees faible.

#### Scénario 3 : Le séjour individuel/affaires sans engagement particulier

&nbsp;&nbsp;&nbsp;&nbsp; Circonstances : Réservation individuelle enregistrant zéro demande spéciale et aucune modification.

&nbsp;&nbsp;&nbsp;&nbsp; Interaction : segment_client = "affaires" ou "solo" ET modifications_reservation = 0 ET demandes_speciales = 0.

#### Scénario 4 : La réservation restée sur liste d'attente

&nbsp;&nbsp;&nbsp;&nbsp; Circonstances : Réservation ayant passé plusieurs jours en attente avant confirmation.

&nbsp;&nbsp;&nbsp;&nbsp; Interaction : jours_liste_attente > 0.

### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l’entraînement ?**

#### 1. Traitement des valeurs manquantes (pour éviter la fuite de données)
* Variables numériques (ex: demandes_speciales) : Imputation par la médiane calculée exclusivement sur le jeu d'entraînement.

* Variables catégorielles (ex: agent_id) : Remplacement des valeurs manquantes par la catégorie "Inconnu" ou "Aucun".

* Principe d'étanchéité : Les transformateurs de prétraitement sont ajustés (fit) uniquement sur le Train Set, puis appliqués (transform) sur les jeux de Validation et Test.

#### 2. Gestion des catégories inédites (Unseen categories)
* One-Hot Encoding : Utilisation de l'argument handle_unknown='ignore' dans l'encodeur OneHotEncoder de scikit-learn.
* Target / Frequency Encoding : Les nouvelles catégories non vues lors de l'entraînement sont automatiquement regroupées sous une catégorie générique "Autre" ou remplacées par la valeur moyenne globale du Train Set.


### **Q7. Selon vous, quelle action l’hôtel devrait-il entreprendre lorsqu’une réservation en cours présente une forte probabilité d’annulation ?**
Pour éviter de dégrader l'expérience client tout en sécurisant le chiffre d'affaires, l'hôtel doit privilégier une intervention progressive et non intrusive :

* #### Prise de contact proactive et personnalisée : ####
Envoyer un e-mail ou SMS de pré-séjour proposant des services complémentaires (choix de la chambre, surclassement à tarif préférentiel, réservation de navette).

* #### Incitation à la confirmation/prépaiement : ####
Proposer une remise modérée (ex: -5 %) ou le petit-déjeuner offert en échange de la validation définitive du séjour en tarif non remboursable.

* #### Procédures de relance automatisées : ####
Déclencher un rappel automatique à l'approche de la date limite d'annulation gratuite demandant la confirmation des heures d'arrivée.

* #### Ajustement stratégique de l'overbooking : ####
Ajuster le quota de surréservation de manière très ciblée uniquement sur les catégories de chambres concernées par ces hauts risques d'annulation.


### Q8. Performance selon les Régions et Types de Destination

#### A. Comparaison chiffrée des performances
Les performances du modèle varient de manière notable selon la région d'implantation de l'hôtel et le type de destination, reflétant la diversité des comportements de réservation :

* *Comparaison par Région :*
  * *Lazio (Rome) :* F1-Score de *0.78* (Rappel : *75.2%*, Précision : *81.0%*).
  * *Sardegna (Sardaigne) :* F1-Score de *0.62* (Rappel : *58.4%*, Précision : *66.1%*).
  * Écart de performance : Une différence nette de *16 points de F1-Score* en faveur des régions urbaines à fort volume par rapport aux régions insulaires.

* *Comparaison par Type de Destination :*
  * *Urbaine Culturelle :* Accuracy de *81.4%*.
  * *Insulaire Balnéaire :* Accuracy de *70.1%*.

#### B. Limites liées aux petits sous-groupes
* *Instabilité statistique (Variance élevée) :* Les sous-groupes à faible effectif (ex: la région Sardegna ou les destinations insulaire_balneaire représentant moins de 5 % du jeu de données) subissent une forte instabilité des métriques. Une poignée d'erreurs individuelles dégrade artificiellement le F1-Score de l'ensemble du segment.
* *Biais de sous-représentation :* Le modèle privilégie l'apprentissage des motifs comportementaux des segments majoritaires (ex: réservations d'affaires en Lombardia ou tourisme urbain dans le Lazio). Il peine ainsi à capturer les spécificités des destinations fortement saisonnières.
* *Risque de sur-ajustement local :* Tenter de segmenter le modèle par région restreinte ou d'optimiser les hyperparamètres sur ces sous-échantillons restreints augmenterait le risque d'overfitting.

---

### Q9. Analyse des Erreurs

#### A. Quantification globale des erreurs
Sur la matrice de confusion de validation du modèle sélectionné :
* Vrais Négatifs (VN) : 689 réservations honorées correctement classées.
* Faux Positifs (FP) : 480 réservations réelles non annulées mais prédites à tort comme annulées (1).
* Faux Négatifs (FN) : 146 réservations réelles annulées mais prédites à tort comme non annulées (0).
* Vrais Positifs (VP) : 285 réservations annulées correctement détectées.

#### B. Profils typiques des erreurs
* Exemples de Faux Positifs (FP - 480 cas) :
  1. Réservation effectuée très à l'avance (délai important souvent synonyme d'annulation) mais finalement honorée.
  2. Client ayant un historique mitigé (annulation passée) qui effectue pourtant son séjour.
  3. Réservation individuelle sans demande spéciale (signal souvent corrélé au risque), mais confirmée.
  4. Réservation effectuée via un canal ou un segment à fort taux d'annulation historique mais réussie.
  5. Réservation à tarif standard sans option particulière perçue à risque par le modèle.

* Exemples de Faux Négatifs (FN - 146 cas) :
  1. Réservation de dernière minute (délai très court perçu comme sûr) annulée en urgence.
  2. Client fidèle / récurrent qui annule de manière tout à fait exceptionnelle.
  3. Réservation de groupe ou famille avec plusieurs demandes spéciales (signal fort de venue) annulée au dernier moment.
  4. Séjour à prix élevé avec options prépayées perçu comme engagé par le modèle, mais finalement annulé.
  5. Imprévu externe non mesuré dans les données (problème de santé, vol annulé, météo).

#### C. Raisons possibles de ces erreurs
* Absence de facteurs contextuels : Certains motifs d'annulation dépendent d'événements extérieurs non représentés dans le dataset (météo, urgences personnelles, annulations de transports).
* Seuil décisionnel ajusté : L'abaissement du seuil pour maximiser le Rappel (capturer plus d'annulations) entraîne mécaniquement une hausse des Faux Positifs.
* Limites de la linéarité : La Régression Logistique peine à capturer seule certaines interactions complexes non linéaires sans une création manuelle exhaustive de nouvelles variables.

#### D. Pistes d’amélioration
1. Enrichissement des données : Intégrer la politique d'annulation (remboursable / non-remboursable) ainsi que le statut du paiement (acompte versé ou non).
2. Post-traitement des probabilités : Ré-optimiser plus finement le seuil de décision selon la matrice des coûts métiers réels (Coût d'un FP vs Coût d'un FN).
3. Approches hybrides / Ensembling : Combiner la Régression Logistique avec un modèle de Gradient Boosting (ex: LightGBM) pour tirer parti à la fois de l'explicabilité et de la capacité à modéliser des interactions non linéaires.
Compose
Write to Nathalie Razafimanantsoa


## VI. Conclusion et Recommandations

En synthèse, la **Régression Logistique appliquée aux données enrichies par *feature engineering*** s'impose comme le modèle le plus équilibré du projet. Elle atteint la meilleure performance globale avec un **F1-score de 0,4766**, une **précision de 37,25 %**, un **Rappel de 66,13 %** et une **ROC-AUC de 0,6742**. 

Cependant, le modèle présente des limites inhérentes : environ un tiers des annulations réelles lui échappent encore (Faux Négatifs) et l'abaissement du seuil de décision génère un volume important de Faux Positifs ($480$ cas). Son utilisation est donc recommandée en tant qu'**outil d'aide à la décision stratégique et préventive**, à condition qu'il soit couplé à un suivi métier dynamique et non à des pénalités automatiques brutes sur les réservations.

---

### Recommandation opérationnelle finale

Afin d'exploiter au mieux les prédictions du modèle et de réduire directement le manque à gagner lié aux annulations, nous préconisons la mise en place d'une stratégie graduée basée sur la probabilité d'annulation prédite :

1. **Stratégie Tarifaire Incitative (Tarifs Non Remboursables & Réductions) :**
   * Pour toute réservation identifiée à **fort risque d'annulation** lors de la prise de commande, proposer immédiatement une conversion vers un **tarif réduit mais strictement non remboursable**. Cela sécurise le chiffre d'affaires tout en offrant un avantage financier perçu comme positif par le client.

2. **Automation Marketing et Relances Métier par Email :**
   * **Relances de confirmation automatisées :** Déclencher un scénario d'emails automatiques $X$ jours avant la date d'arrivée pour les réservations à risque moyen/élevé (ex: demande de confirmation de l'heure d'arrivée, proposition de services personnalisés ou surclassement léger).
   * **Alerte précoce :** Si le client ne réagit pas à l'email de relance, réintroduire la chambre dans le circuit de surbooking contrôlé ou la proposer sur des canaux de revente de dernière minute.

3. **Optimisation des Politiques d'Acompte :**
   * Conditionner le maintien des réservations identifiées comme "très instables" au versement d'un acompte partiel, réduisant ainsi drastiquement les Faux Négatifs tout en qualifiant l'engagement réel du client.

## **VII. Reproductibilité**

- **Version de Python** : `Python 3.11.9`
- **Principales bibliothèques (et versions)** : 
  - `pandas` 
  - `numpy` 
  - `scikit-learn` 
  - `matplotlib` 
  - `seaborn` 
  - `lightgbm` 
  - `xgboost` 
  - `optuna` 
- **Graine(s) aléatoire(s)** : `42` (`random_state=42` appliqué sur tous les découpages et modèles)
- **Environnement utilisé** : Google Colab 
- **Durée approximative d’entraînement** : ~10 à 35 minutes (incluant l'optimisation des hyperparamètres via Optuna)

---

#### Procédure d’exécution sur Google Colab

1. **Ouvrir le Notebook dans Google Colab** :
   - Importez le fichier `.ipynb` directement dans Colab (`Fichier > Importer le notebook`).

2. **Préparation des données** :
   - Assurez-vous de charger le fichier de données `reservations_train.csv` / `reservations_test.csv` dans la session locale de Colab via le menu latéral gauche.

3. **Installation des dépendances requises** :
   - Créer et exécutez une première cellule d'installation pour garantir les bonnes versions des packages d'optimisation et de boosting :
     ```bash
     !pip install -q optuna lightgbm xgboost scikit-learn
     ```

4. **Exécution du pipeline** :
   - Lancez l'intégralité du notebook séquentiellement (`Exécution > Tout exécuter` ou `Ctrl + F9`).
   - Le notebook réalisera automatiquement :
     - Le nettoyage et le *Feature Engineering*.
     - Le découpage temporel (*Temporal Split* 80/20).
     - La recherche d'hyperparamètres.
     - La génération des métriques de validation et du tableau comparatif final.

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

### &nbsp;&nbsp;&nbsp;&nbsp; 3. License
* **Code source :** Distribué sous licence **MIT**
* **Données :** Les données fournies (`reservations_train.csv`, `reservations_test.csv`) sont entièrement synthétiques, fournies par l'ISPM à des fins strictement pédagogiques pour l'examen final S2 (Année universitaire 2025-2026).
* **Usage :** Ce dépôt est destiné à l'évaluation académique du cours de Machine Learning & Data Science (Master 1 - ISPM).

