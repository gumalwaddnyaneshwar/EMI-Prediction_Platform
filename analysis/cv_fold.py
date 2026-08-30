import sys, json, os
sys.path.append('/home/claude/emipredict-ai/src')
import numpy as np, joblib
from train_utils import load_featured_data, fit_or_load_preprocessor, CLF_TARGET
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

fold_idx = int(sys.argv[1])  # 0-4
RESULTS_PATH = '/home/claude/emipredict-ai/analysis/cv5_results.json'

df = load_featured_data()
le = LabelEncoder()
y_all = le.fit_transform(df[CLF_TARGET])
hr_idx = list(le.classes_).index('High_Risk')
preprocessor = fit_or_load_preprocessor(df, refit=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(df, y_all))
tr_idx, te_idx = splits[fold_idx]

X_tr = preprocessor.transform(df.iloc[tr_idx])
X_te = preprocessor.transform(df.iloc[te_idx])
y_tr, y_te = y_all[tr_idx], y_all[te_idx]
sw = np.ones(len(y_tr)); sw[y_tr == hr_idx] = 2.0

m = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=42)
m.fit(X_tr, y_tr, sample_weight=sw)
pred = m.predict(X_te)
acc = accuracy_score(y_te, pred)
f1 = f1_score(y_te, pred, average='macro')

results = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        results = json.load(f)
results[str(fold_idx)] = {'acc': acc, 'f1_macro': f1}
with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Fold {fold_idx}: acc={acc:.4f} f1_macro={f1:.4f}')
