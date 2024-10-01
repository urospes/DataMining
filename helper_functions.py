import pandas as pd
import numpy as np
import imblearn
from sklearn import pipeline
from sklearn import model_selection
from IPython.display import display


# Pomocna funkcija za unakrsnu validaciju modela
def run_cv(model, X, y, pipe, cv, scoring, resampled=False, train_score=False):
    X_c = X.copy(deep=True)
    y_c = y.copy()

    if resampled:
        print("Resampling...")
        oversampler = imblearn.over_sampling.SMOTE(sampling_strategy=0.6, random_state=42, k_neighbors=5, n_jobs=-1)
        undersampler = imblearn.under_sampling.EditedNearestNeighbours(sampling_strategy="all", n_neighbors=5, n_jobs=-1)
        pipe_w_m = imblearn.pipeline.make_pipeline(*list(pipe.named_steps.values()), oversampler, undersampler, model)
    else:
        pipe_w_m = pipeline.make_pipeline(pipe, model)

    results = model_selection.cross_validate(pipe_w_m, X_c, y_c, cv=cv, scoring=scoring, return_train_score=train_score, verbose=1)
    return results


# Pomocna funkcija za pregledniji prikaz rezultata
def display_results(results, scoring, with_std=False):
    res_df = pd.DataFrame()

    for model in results:
        model_res = results[model]
        res_dict = dict()
        for metric in scoring.keys():
            metric_train = f'train_{metric}'
            metric_test = f'test_{metric}'
            res_dict[f'{metric_train}_avg'] =  np.mean(model_res[metric_train])
            res_dict[f'{metric_test}_avg'] =  np.mean(model_res[metric_test])
            if with_std:
                res_dict[f'{metric_train}_std'] =  np.std(model_res[metric_train])
                res_dict[f'{metric_test}_std'] =  np.std(model_res[metric_test])   
        res_df[model] = res_dict

    display(res_df.T)