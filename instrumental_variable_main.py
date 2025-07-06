# %%
import pandas as pd
import mlflow
import numpy as np
from data_preprocessing.init_data_prepro import init_data_preprocessing
from data_preprocessing.data_prepro_func import load_data, load_config

# %%
if __name__ == "__main__":
    
    """
    load data and config
    """
    # Load config
    model_config_path = 'config/model_config.yaml'
    model_config = load_config(model_config_path)

    # Load data config
    data_config_path = 'config/data_config.yaml'
    data_config = load_config(data_config_path)

    # load data
    cars_file_path = cars_file_path = data_config['loading_path_data']['data_cars']
    cars_db        = load_data(cars_file_path)

    # %%
    # preprocessing the data
    preprocess_df, endogenous_var, exogenous_var, added_depvar = init_data_preprocessing(
        data_config, 
        cars_db
        )

    # %%
    from instrument_creation.instrument_creation_functions import *
     # Extract instruments
    Z, instrument_vars = init_instrument_creation(
        preprocess_df, 
        exogenous_var, 
        data_config['instrument_creation']['twodegree_polynomial_instruments']
        )
    
    # %%
    # Finalize the dataframe
    final_df = preprocess_df[
        ['marketid', data_config['var_of_interest']['productvar'], 'state']
        + added_depvar
        + endogenous_var
        + exogenous_var
    ].reset_index(drop=True).join(Z.reset_index(drop=True))

    final_df[['sellingprice','odometer']]=final_df[['sellingprice','odometer']]/10000

    # %%[markdown]
    # ---
    # ---
    # # Le modèle structurel de la demande
    # Sur le marché $t$, le consommateur $i$ choisit entre les voitures de $J$ marques $1, ..., J$ et dispose également d'une option extérieure $0$ (c’est-à-dire ne pas acheter de voiture). L’utilité indirecte qui décrit les préférences du consommateur est donnée par :
    # $$
    # U_{ijt} = X_{jt}^{T}\beta_{x} + \beta_{p} P_{jt} + \xi_{jt} + \epsilon_{ijt} \equiv \delta_{jt} + \epsilon_{ijt}
    # $$
    # - $P_{jt}$ est le prix agrégé des voitures de la marque $j$  
    # - $X_{jt}$ contient les caractéristiques non liées au prix : kilométrage (`odometer`), état, année de fabrication (`year`), taille (`body`), couleurs intérieure (`interior`) et extérieure (`color`)  
    # - $\xi_{jt}$ est la qualité du produit non observée  
    # - $\epsilon_{ijt}$ est l’hétérogénéité non observée entre consommateurs  
    # - $\delta_{jt} \equiv X_{jt}^{T}\beta_{x} + \beta_{p} P_{jt} + \xi_{jt}$ est l’utilité moyenne du produit $j$  
    # - L’utilité moyenne de l’option extérieure est normalisée à $0$
    # En supposant que $\{\epsilon_{ijt}\}_{ijt}$ soient i.i.d. et suivent une loi de Gumbel, la probabilité que le consommateur $i$ achète le produit $j$ est donnée par la loi logistique multinomiale :
    # $$
    # \text{Prob}[Choix = j \mid X_{t}, P_{t}] = \frac{\exp(\delta_{jt})}{1 + \sum_{l=1}^{J} \exp(\delta_{lt})}
    # $$
    # On peut montrer que :
    # $$
    # \log\left(\text{Prob}[Choix = j \mid X_{t}, P_{t}]\right) - \log\left(\text{Prob}[Choix = 0 \mid X_{t}, P_{t}]\right) = \delta_{jt}
    # $$
    # Comme les probabilités de choix correspondent aux parts de marché $S_{jt}$ en équilibre, et puisque $\delta_{jt} \equiv X_{jt} \beta_{x} + \beta_{p} P_{jt} + \xi_{jt}$, on obtient la relation de demande suivante :
    # $$
    # \log(S_{jt}) - \log(S_{0t}) = X_{jt} \beta_{x} + \beta_{p} P_{jt} + \xi_{jt} \hspace{10mm} (1)
    # $$
    # Où $S_{jt}$ est la part de marché du produit $j$ au temps $t$, et $S_{0t}$ est la part de marché de l’option extérieure (ne pas acheter de voiture).
    # C’est cette équation que l’on souhaite estimer. (Voir Berry, S. 1994, pour plus de détails sur ce modèle.)
    # ---
    # ### Prix endogène et estimation par variables instrumentales
    # Le prix étant endogène, il faut utiliser des méthodes basées sur des variables instrumentales $Z_{jt}$ pour identifier l’effet causal de $P_{jt}$ sur la demande (i.e. $\beta_{p}$). Une méthode de base utilisée dans ce contexte est la méthode des **doubles moindres carrés (2SLS)**, qui consiste à estimer :
    # 1. Une spécification linéaire de la relation :
    # $$
    # P_{jt} = \mathbb{E}[P_{jt} \mid X_{t}, Z_{jt}] + \nu_{jt} \hspace{10mm} (2)
    # $$
    # Elle est linéaire si :
    # $$
    # \mathbb{E}[P_{jt} \mid X_{jt}, Z_{jt}] = X_{jt}^{T}\gamma_1 + Z_{jt}^{T}\gamma_2
    # $$
    # On note $\hat{P}_{jt}$ la prédiction du prix obtenue.
    # 2. Puis on estime :
    # $$
    # \log(S_{jt}) - \log(S_{0t}) = X_{jt} \beta_{x} + \beta_{p} \hat{P}_{jt} + \xi_{jt} \hspace{10mm} (3)
    # $$
    # ---
    # ### Problèmes potentiels de spécification du modèle et utilité des algorithmes d’apprentissage automatique :
    # - Sélection des variables dans $X$ lorsque leur nombre est trop élevé par rapport à la taille de l’échantillon (ce n’est pas notre cas ici, mais on testera l’approche)
    # - Sélection des variables dans $X$ et $Z$ dans l'équation du prix lorsque leur nombre est élevé par rapport à la taille de l’échantillon (c’est le cas ici si on construit les instruments à partir de fonctions polynomiales d’ordre 2 des caractéristiques des produits concurrents — voir l’option `twodegree_polynomial_instruments=False` dans la fonction `aggregate_data`)
    # - Approximation plus flexible de $\mathbb{E}[P_{jt} \mid X_{t}, Z_{jt}]$ pour relâcher l’hypothèse de linéarité, en utilisant des mesures de performance hors échantillon

    # %%[markdown]
    # ---
    # ---
    # # Estimation de l'equation (3) par MCO sans remplacer $P_{jt}$ par $\hat{P}_{jt}$

    # Voici le probleme de moindres carres a resoudre:

    #$$\min_{\beta} \sum_{j,t}\left(log(S_{jt})-log(S_{0t}) - X_{jt}\beta_{x} - \beta_{p} P_{jt}\right)^{2}$$
    # %%
    # Run simple OLS regression on the final_df
    from models_IV.OLS import LinearModel
   
    OLS_base = LinearModel(
        target_col='log_share_ratio',
        feature_cols=endogenous_var + exogenous_var,
        test_size=model_config['base_model']['test_size'],
        random_state=model_config['base_model']['random_state'],
    )

    # %%
    OLS_base._train_test_split(
        final_df
    )
    # %%# %%
    
    OLS_base.train(
        OLS_base.X_train,
        OLS_base.y_train
    )
   # %% 
    prediction = OLS_base._predict(
        OLS_base.X_test, 
        OLS_base.feature_cols
    )
    # %%
    OLS_base.evaluate(
        OLS_base.y_test, 
        prediction
    )
 
    OLS_base.compute_covariance_matrix(   
        OLS_base.X_train,
        OLS_base.y_train
    )
    
    OLS_base.regression_summary(
        OLS_base.model,
        OLS_base.X_test,
        OLS_base.y_test,
        feature_names=OLS_base.feature_cols
    )

    OLS_base.plot_true_predicted(
        OLS_base.y_test, prediction,
        ModelName="OLS", VarName="log_share_ratio"
        )


    # %%[markdown]
    # Selection des variables de X, Z dans l'equation du prix (2)
    #Un des motifs pour recourir à une telle sélection basée sur les données est la présence d'un trop grand nombre  de variables exogènes dans l'équation du prix par rapport à la taille de l'échantillon. 
    #Voici le probleme de moindres carrés regularizés lorsqu'on recours à une regularisation d'elastic-net:
    #$$\min_{\gamma} \sum_{j,t} \left(P_{jt} - (X_{jt}^{T}, Z_{jt}^{T})\gamma \right)^{2} + \lambda \left(\alpha \sum_{k}|\gamma_{k}| +  (1-\alpha)\sum_{k}\gamma_{k}^{2}\right)$$
    #Les regularisation de Lasso ($\alpha=1$) et Ridge ($\alpha=0$) sont des cas particuliers
    # %%
    from models_IV.lasso_cv_model import LassoCVModel2
    
    lasso_model = LassoCVModel2(
        target_col='sellingprice',
        feature_cols=exogenous_var + instrument_vars,
        unpenalized_cols=exogenous_var,
        test_size=model_config['base_model']['test_size'],
        random_state=model_config['base_model']['random_state'],
        # parameters for LassoCV
        cv=model_config['cv_lasso']['cv'],
        alphas=np.power(2.0, np.linspace(-15, 1, 100)),
        max_iter=model_config['cv_lasso']['max_iter'],
        n_jobs=model_config['cv_lasso']['n_jobs']
    )
    # %%
    lasso_model._train_test_split(final_df)

    lasso_model.train(lasso_model.X_train, lasso_model.y_train)
    
    metrics = lasso_model.evaluate(lasso_model.X_train, lasso_model.y_train, lasso_model.X_test, lasso_model.y_test)
    print(metrics)
    lasso_model.plot_cv_path()
    print(lasso_model.coefs)

    prediction_lasso = lasso_model._predict(lasso_model.X_test)
    lasso_model.plot_true_predicted(
        lasso_model.y_test, prediction_lasso,
        ModelName="LassoCV", VarName="sellingprice"
    )


    # %%

    from models_IV.RandomForest import RandomForestModel
    
    rf_model = RandomForestModel(
        target_col='sellingprice',
        feature_cols=exogenous_var + instrument_vars,
        test_size=model_config['base_model']['test_size'],
        random_state=model_config['base_model']['random_state'],
        # parameters for RandomForestRegressor
        n_estimators=model_config['randomforest']['n_estimators'],
        max_depth=model_config['randomforest']['max_depth'],
        min_samples_split=model_config['randomforest']['min_samples_split'],
        min_samples_leaf=model_config['randomforest']['min_samples_leaf'],
        bootstrap=model_config['randomforest']['bootstrap'],
        n_jobs=model_config['randomforest']['n_jobs']
    )
    
    rf_model._train_test_split(final_df)
    # %%
    rf_model.train(rf_model.X_train, rf_model.y_train)

    metrics = rf_model.evaluate(
        rf_model.X_train, rf_model.y_train, 
        rf_model.X_test, rf_model.y_test
        )
    print(metrics)
    prediction_rf = rf_model._predict(rf_model.X_test)
    rf_model.plot_true_predicted(
        rf_model.y_test, prediction_rf,
        ModelName="RandomForest", VarName="sellingprice"
    )


    # %%
    from models_IV.XGBoost import XGBoostModel
    xgb_model = XGBoostModel(
        target_col='sellingprice',
        feature_cols= exogenous_var + instrument_vars,
        test_size=model_config['base_model']['test_size'],
        random_state=model_config['base_model']['random_state'],
        # parameters for XGBRegressor
        eval_metric=model_config['xgboost']['eval_metric'],
        n_estimators=model_config['xgboost']['n_estimators'],
        max_depth=model_config['xgboost']['max_depth'],
        learning_rate=model_config['xgboost']['learning_rate'],
        subsample=model_config['xgboost']['subsample'],
        colsample_bytree=model_config['xgboost']['colsample_bytree'],
        use_label_encoder=model_config['xgboost']['use_label_encoder']
        )
    
    xgb_model._train_test_split(final_df)
    # %%
    xgb_model.train(xgb_model.X_train, xgb_model.y_train)
    metrics = xgb_model.evaluate(
        xgb_model.X_train, xgb_model.y_train,
        xgb_model.X_test, xgb_model.y_test
        )
    print(metrics)
    prediction_xgb = xgb_model._predict(xgb_model.X_test)
    xgb_model.plot_true_predicted(
        xgb_model.y_test, prediction_xgb,
        ModelName="XGBoost", VarName="sellingprice"
        )

    # %%[markdown]
    # ---
    # ---
    # # Estimation de l'equation de la demande par MCO avec remplacement de $P_{jt}$ par $\hat{P}_{jt}$

    # %%
    # Run demand regression by simple OLS on the final_df
    from models_IV.OLS import LinearModel
    from model_evaluation.evaluation_functions_2 import *
    # %%
    IV_linear_demand_model = LinearModel(
        target_col='log_share_ratio',
        feature_cols=endogenous_var + exogenous_var + instrument_vars,
        test_size=model_config['base_model']['test_size'],
        random_state=model_config['base_model']['random_state'],
    )

    IV_linear_demand_model._train_test_split(
        final_df
    )
    # %%
    # Extract training and test regressors for the first stage model (i.e. price prediction model) and the second stage model (demand model). Then update the feature_cols for the IV_linear_demand_model.
    first_stage_model_X_train = IV_linear_demand_model.X_train[exogenous_var + instrument_vars ]
    first_stage_model_X_test = IV_linear_demand_model.X_test[exogenous_var + instrument_vars]
    IV_linear_demand_model.X_train = IV_linear_demand_model.X_train[endogenous_var + exogenous_var]
    IV_linear_demand_model.X_test = IV_linear_demand_model.X_test[endogenous_var + exogenous_var]
    IV_linear_demand_model.feature_cols = endogenous_var + exogenous_var
    # %% 
    # Get coefs and willingness to pay values and their respective p-values
    IV_linear_demand_model, demand_coefs, demand_target_pred = demand_evaluation(
        IV_linear_demand_model,
        exogenous_prediction=pd.Series(lasso_model._predict(first_stage_model_X_train), name=lasso_model.target_col)
        )
        # exogenous_prediction=rf_model._predict(rf_model.X_train),
   # %% 
   # plot coefficients and willingness to pay values
    plot_coefs(demand_coefs['coef'],demand_coefs['coef-pvalue'], titlename='Linear Demand Coefficients')
    plot_coefs(demand_coefs['wtp'], demand_coefs['wtp-pvalue'], titlename='Willingness to Pay (WTP) Coefficients')
     # %%   

    # %% [markdown]
    # ---
    # ---
    # # Réduction de dimension des prédicteurs du prix: Analyse en Composantes Principales, ACP (PCA en anglais)
    # 
    # On applique l'ACP pour réduire la dimension des prédicteurs du prix. La réduction du nombre de variables pourrait nous donner plus de flexibilité dans la prédiction du prix.
    # 
    # Soit $U$ la matrice des prédicteurs du prix (par exemple les variables $X$ et $Z$). L'analyse en composantes principales (ACP) consiste à trouver une matrice orthonormée $P$ telle que $V = UP$ soit une transformation linéaire de $U$ vers un nouvel espace dans lequel les composantes sont non corrélées et ordonnées par variance décroissante.
    # 
    # La matrice $V$ est appelée matrice des composantes principales. Si l'on conserve les $v$ premières colonnes de $V$, on obtient une représentation de $U$ dans un espace de dimension réduite qui préserve le plus possible la variance initiale. Cette représentation est optimale au sens où aucune autre combinaison de $v$ vecteurs quelconques ne permet de conserver davantage de variance.


    # %%

    from models_IV.PCA import PCAWrapper
    
    pca_model = PCAWrapper(
        variance_ratio_preserved=0.95, 
        test_size=model_config['base_model']['test_size'],
        feature_cols=instrument_vars + exogenous_var
        )

    pca_model._train_test_split(final_df)

    # %%
    pca_model.fit(pca_model.X_train)

    # (Optional) Plot variance explained
    pca_model.plot_variance_explained()


    # %%[markdown]
    #---
    #### Régression de la demande après remplacement du prix par sa prédiction exogène obtenue par regression Lasso
    # %%
    from model_evaluation.evaluation_functions import *

    price_train_predicted = pd.Series(
        OLS_base._predict(OLS_base.X_train, OLS_base.feature_cols),
          name=endogenous_var[-1]
        )
    
    X_train = pd.DataFrame(OLS_base.X_train, columns=OLS_base.feature_cols)

    Y_train = pd.Series(OLS_base)
    

    model, coefs, _ = regress_demand(data_train[endogvars+exogvars], data_train[depvars[-1]], price_train_predicted)


    # %%
    # Set the MLflow tracking URI to localhost with the desired port (e.g., 5000)
    import mlflow
    from mlflow.models import infer_signature
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd

    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    
    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Quickstart")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )

    
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    iris_feature_names = datasets.load_iris().feature_names

    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    result[:4]


        
