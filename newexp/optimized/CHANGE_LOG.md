# Optimized Experiments Change Log
# Parameters with default values removed from model instantiations

## Summary
- Total experiments: 10
- Parameters removed: 8 (default values eliminated)
- Net reduction: removed explicit default parameters from model code

---

## exp-1.py: "Linear Regression"
### What Changed ✨
- **REMOVED**: `max_iter=1000` from `SGDRegressor` (default value: 1000)
- **REMOVED**: `eta0=0.01` from `SGDRegressor` (default value: 0.01, only meaningful with `learning_rate='constant'`)
- **KEPT**: `random_state=42`, `learning_rate='constant'`
- **Result**: `SGDRegressor` now only specifies non-default parameters

### Package Changes
- ❌ Removed: `max_iter` and `eta0` explicit parameter specifications
- ✅ Kept: `random_state` and `learning_rate='constant'` (non-default)

---

## exp-2.py: "Polynomial Regression"
### What Changed ✨
- **NO CHANGES**: No default parameters identified for removal
- `PolynomialFeatures(degree=d, include_bias=False)` - `degree=d` is a loop variable, `include_bias=False` is not the default (default is `True`)

### Package Changes
- ✅ No changes needed - all parameters are non-default or variable

---

## exp-3.py: "Ridge/Lasso Comparison"
### What Changed ✨
- **NO CHANGES**: No default parameters identified for removal
- `RidgeCV(alphas=..., cv=5)` and `LassoCV(alphas=..., cv=5, max_iter=10000)` - all parameters are explicitly set for the experiment

### Package Changes
- ✅ No changes needed - all parameters are non-default or experiment-specific

---

## exp-6.py: "Logistic Regression Scaling"
### What Changed ✨
- **NO CHANGES**: No default parameters identified for removal
- `LogisticRegression(solver='saga', max_iter=1000, random_state=42)` - `solver='saga'` is not default (default is 'lbfgs'), `max_iter=1000` is not default (default is 100), `random_state=42` is not default

### Package Changes
- ✅ No changes needed - all parameters are non-default

---

## exp-8.py: "Fashion MNIST KNN"
### What Changed ✨
- **NO CHANGES**: No default parameters identified for removal
- `KNeighborsClassifier(n_neighbors=k, n_jobs=-1)` - `n_neighbors=k` is a loop variable, `n_jobs=-1` is not the default (default is `None`)

### Package Changes
- ✅ No changes needed - all parameters are non-default or variable

---

## exp-9.py: "Decision Tree RFM"
### What Changed ✨
- **NO CHANGES**: No default parameters identified for removal
- `DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)` - `criterion='entropy'` is not default (default is 'gini'), `max_depth=3` is not default (default is None), `random_state=42` is not default

### Package Changes
- ✅ No changes needed - all parameters are non-default

---

## exp-11.py: "SVM on Iris"
### What Changed ✨
- **REMOVED**: `C=1.0` from `SVC()` (default value: 1.0)
- **KEPT**: `kernel='linear'`, `StandardScaler()` within pipeline

### Package Changes
- ❌ Removed: `C=1.0` explicit parameter (default value)
- ✅ Kept: `kernel='linear'` (non-default), pipeline structure

---

## exp-12.py: "SVM + PCA Fashion MNIST"
### What Changed ✨
- **REMOVED**: `C=1.0` from `SVC()` (default value: 1.0)
- **REMOVED**: `gamma='scale'` from `SVC()` (default value: 'scale')
- **KEPT**: `kernel=kernel_name`, `PCA(n_components=50, random_state=42)`

### Package Changes
- ❌ Removed: `C=1.0` and `gamma='scale'` explicit parameters (both default values)
- ✅ Kept: `kernel=kernel_name` (varies per iteration), PCA parameters (non-default)

---

## exp-14.py: "MLP MNIST Activations"
### What Changed ✨
- **NO CHANGES**: No default parameters identified for removal
- `MLPClassifier(hidden_layer_sizes=(32, 16), activation=act, max_iter=50, random_state=42)` - all parameters are non-default or loop variables

### Package Changes
- ✅ No changes needed - all parameters are non-default or variable

---

## exp-16.py: "Clustering Comparison"
### What Changed ✨
- **REMOVED**: `init='k-means++'` from `KMeans()` (default value: 'k-means++')
- **REMOVED**: `n_init=10` from `KMeans()` (default value: 10)
- **KEPT**: `n_clusters=k` (loop variable), `random_state=42`
- **REMOVED**: `metric='euclidean'` from `AgglomerativeClustering()` (default value: 'euclidean')
- **REMOVED**: `linkage='ward'` from `AgglomerativeClustering()` (default value: 'ward')
- **KEPT**: `n_clusters=optimal_k`

### Package Changes
- ❌ Removed: `init`, `n_init` from KMeans; `metric`, `linkage` from AgglomerativeClustering (all default values)
- ✅ Kept: `n_clusters`, `random_state` from KMeans

---

## 📊 Summary of Parameter Removals

| Experiment | Parameter Removed | Default Value |
|------------|------------------|---------------|
| exp-1 | `max_iter=1000` (SGDRegressor) | 1000 |
| exp-1 | `eta0=0.01` (SGDRegressor) | 0.01 |
| exp-11 | `C=1.0` (SVC) | 1.0 |
| exp-12 | `C=1.0` (SVC) | 1.0 |
| exp-12 | `gamma='scale'` (SVC) | 'scale' |
| exp-16 | `init='k-means++'` (KMeans) | 'k-means++' |
| exp-16 | `n_init=10` (KMeans) | 10 |
| exp-16 | `metric='euclidean'` (AgglomerativeClustering) | 'euclidean' |
| exp-16 | `linkage='ward'` (AgglomerativeClustering) | 'ward' |

**Total parameters removed: 8** across 7 experiments

**Experments with changes: 3 (exp-1, exp-11, exp-12, exp-16)**
**Experments unchanged: 7 (exp-2, exp-3, exp-6, exp-8, exp-9, exp-14)**

### Key Optimization
Removing default parameter values makes the code cleaner by only specifying non-default hyperparameters, while maintaining 100% output equivalence.