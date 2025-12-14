"""Full comparison harness between myXGBoost and official xgboost.

Runs repeated trials for regression and classification with hyperparameter parity,
saves per-run prediction CSVs (including true targets), aggregates results,
and creates residual/ROC plots when matplotlib is available.
"""
import time
import csv
import os
from datetime import datetime
import numpy as np

try:
    from sklearn.datasets import make_regression, make_classification
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    raise RuntimeError("sklearn is required to run the full comparison harness")

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from myXGBoost.estimators.regressor import XGBRegressor as MyXGBReg
from myXGBoost.estimators.classifier import XGBClassifier as MyXGBClf

def try_import_xgboost():
    try:
        from xgboost import XGBRegressor, XGBClassifier
        return XGBRegressor, XGBClassifier
    except Exception:
        return None, None


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def run_regression(n_runs=3, n_samples=5000, n_features=30, n_estimators=100, out_dir='regression_results', random_state=42):
    ensure_dir(out_dir)
    XGBReg, _ = try_import_xgboost()
    if XGBReg is None:
        raise RuntimeError('xgboost not installed')

    summary_rows = []
    rng = np.random.RandomState(random_state)
    for run in range(n_runs):
        seed = rng.randint(0, 2**31 - 1)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=seed)
        split = int(n_samples * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        params = dict(n_estimators=n_estimators, random_state=seed)

        # myXGBoost
        my = MyXGBReg(**params)
        t0 = time.time(); my.fit(X_train, y_train); t1 = time.time()
        y_my = my.predict(X_test)
        my_time = t1 - t0

        # xgboost
        xgb = XGBReg(**params)
        t0 = time.time(); xgb.fit(X_train, y_train); t1 = time.time()
        y_xgb = xgb.predict(X_test)
        xgb_time = t1 - t0

        # metrics
        my_rmse = float(np.sqrt(mean_squared_error(y_test, y_my)))
        xgb_rmse = float(np.sqrt(mean_squared_error(y_test, y_xgb)))
        my_mae = float(mean_absolute_error(y_test, y_my))
        xgb_mae = float(mean_absolute_error(y_test, y_xgb))
        my_r2 = float(r2_score(y_test, y_my))
        xgb_r2 = float(r2_score(y_test, y_xgb))

        # prediction similarity (small slice)
        small_n = min(100, len(X_test))
        X_small = X_test[:small_n]
        y_true_small = y_test[:small_n]
        y_my_small = y_my[:small_n]
        y_xgb_small = y_xgb[:small_n]
        pred_rmse = float(np.sqrt(mean_squared_error(y_my_small, y_xgb_small)))
        pred_corr = float(np.corrcoef(y_my_small, y_xgb_small)[0, 1])

        # save per-run predictions (with true targets)
        tstamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        preds_fname = os.path.join(out_dir, f'preds_regression_run{run+1}_{tstamp}.csv')
        stacked = np.column_stack([y_test, y_my, y_xgb])
        np.savetxt(preds_fname, stacked, delimiter=',', header='y_true,myxgboost,xgboost', comments='')

        # residual plot
        if _HAS_MPL:
            plt.figure(figsize=(6, 4))
            plt.scatter(y_test, y_test - y_my, alpha=0.4, label='myxgboost')
            plt.scatter(y_test, y_test - y_xgb, alpha=0.4, label='xgboost')
            plt.axhline(0, color='k', linewidth=0.8)
            plt.xlabel('True target'); plt.ylabel('Residual (y - y_pred)'); plt.legend()
            plot_fname = os.path.join(out_dir, f'residuals_regression_run{run+1}_{tstamp}.png')
            plt.savefig(plot_fname, bbox_inches='tight')
            plt.close()
        else:
            plot_fname = None

        row = {
            'run': run + 1,
            'seed': seed,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_estimators': n_estimators,
            'my_time': my_time,
            'my_rmse': my_rmse,
            'my_mae': my_mae,
            'my_r2': my_r2,
            'xgb_time': xgb_time,
            'xgb_rmse': xgb_rmse,
            'xgb_mae': xgb_mae,
            'xgb_r2': xgb_r2,
            'predictions_rmse': pred_rmse,
            'predictions_corr': pred_corr,
            'predictions_file': preds_fname,
            'residuals_plot': plot_fname,
        }
        summary_rows.append(row)

    # write aggregate per-run CSV
    out_summary = os.path.join(out_dir, 'regression_runs.csv')
    keys = list(summary_rows[0].keys())
    with open(out_summary, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def run_classification(n_runs=3, n_samples=5000, n_features=30, n_estimators=100, out_dir='classification_results', random_state=42):
    ensure_dir(out_dir)
    _, XGBClf = try_import_xgboost()
    if XGBClf is None:
        raise RuntimeError('xgboost not installed')

    summary_rows = []
    rng = np.random.RandomState(random_state)
    for run in range(n_runs):
        seed = rng.randint(0, 2**31 - 1)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=min(10, n_features), random_state=seed)
        split = int(n_samples * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        params = dict(n_estimators=n_estimators, random_state=seed, use_label_encoder=False)

        # myXGBoost classifier
        my = MyXGBClf(n_estimators=n_estimators, random_state=seed)
        t0 = time.time(); my.fit(X_train, y_train); t1 = time.time()
        y_my_proba = my.predict_proba(X_test)
        if y_my_proba.ndim > 1:
            y_my_pos = y_my_proba[:, 1]
        else:
            y_my_pos = y_my_proba
        y_my_label = my.predict(X_test)
        my_time = t1 - t0

        # xgboost classifier
        xgb = XGBClf(n_estimators=n_estimators, random_state=seed, use_label_encoder=False)
        t0 = time.time(); xgb.fit(X_train, y_train); t1 = time.time()
        y_xgb_proba = xgb.predict_proba(X_test)[:, 1]
        y_xgb_label = xgb.predict(X_test)
        xgb_time = t1 - t0

        # metrics
        my_acc = float(accuracy_score(y_test, y_my_label))
        xgb_acc = float(accuracy_score(y_test, y_xgb_label))
        my_logloss = float(log_loss(y_test, y_my_pos))
        xgb_logloss = float(log_loss(y_test, y_xgb_proba))
        try:
            my_auc = float(roc_auc_score(y_test, y_my_pos))
        except Exception:
            my_auc = float('nan')
        try:
            xgb_auc = float(roc_auc_score(y_test, y_xgb_proba))
        except Exception:
            xgb_auc = float('nan')

        # save per-run predictions with true targets
        tstamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        preds_fname = os.path.join(out_dir, f'preds_classification_run{run+1}_{tstamp}.csv')
        stacked = np.column_stack([y_test, y_my_pos, y_xgb_proba, y_my_label, y_xgb_label])
        np.savetxt(preds_fname, stacked, delimiter=',', header='y_true,my_prob,xgb_prob,my_label,xgb_label', comments='')

        # ROC plot
        if _HAS_MPL:
            try:
                from sklearn.metrics import roc_curve
                fpr_my, tpr_my, _ = roc_curve(y_test, y_my_pos)
                fpr_x, tpr_x, _ = roc_curve(y_test, y_xgb_proba)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr_my, tpr_my, label='myxgboost')
                plt.plot(fpr_x, tpr_x, label='xgboost')
                plt.plot([0, 1], [0, 1], '--', color='gray')
                plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('ROC')
                plot_fname = os.path.join(out_dir, f'roc_classification_run{run+1}_{tstamp}.png')
                plt.savefig(plot_fname, bbox_inches='tight')
                plt.close()
            except Exception:
                plot_fname = None
        else:
            plot_fname = None

        row = {
            'run': run + 1,
            'seed': seed,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_estimators': n_estimators,
            'my_time': my_time,
            'my_acc': my_acc,
            'my_logloss': my_logloss,
            'my_auc': my_auc,
            'xgb_time': xgb_time,
            'xgb_acc': xgb_acc,
            'xgb_logloss': xgb_logloss,
            'xgb_auc': xgb_auc,
            'predictions_file': preds_fname,
            'roc_plot': plot_fname,
        }
        summary_rows.append(row)

    out_summary = os.path.join(out_dir, 'classification_runs.csv')
    keys = list(summary_rows[0].keys())
    with open(out_summary, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def summarize_and_save(summary_rows, out_path):
    # compute mean/std for numeric columns
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    numeric = df.select_dtypes(include=[float, int])
    stats = numeric.agg(['mean', 'std']).T
    stats.to_csv(out_path)


if __name__ == '__main__':
    # default run settings (adjustable)
    REG_OUT = 'regression_results'
    CLF_OUT = 'classification_results'
    runs = 3
    try:
        reg_rows = run_regression(n_runs=runs, n_samples=5000, n_features=30, n_estimators=100, out_dir=REG_OUT)
        clf_rows = run_classification(n_runs=runs, n_samples=5000, n_features=30, n_estimators=100, out_dir=CLF_OUT)

        # write aggregated stats (requires pandas)
        try:
            summarize_and_save(reg_rows, os.path.join(REG_OUT, 'regression_summary_stats.csv'))
            summarize_and_save(clf_rows, os.path.join(CLF_OUT, 'classification_summary_stats.csv'))
            print('Wrote summary stats (mean/std) for regression and classification')
        except Exception as e:
            print('Could not write summary stats (pandas required):', e)

        print('Completed full comparison. See folders:', REG_OUT, CLF_OUT)
    except Exception as e:
        print('Comparison failed:', e)
