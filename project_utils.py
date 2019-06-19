# Imports
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Impute missing values
from sklearn.impute import SimpleImputer

# Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Models
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from hyperopt import hp, tpe, fmin, Trials, space_eval, STATUS_OK
from hyperopt.pyll.base import scope
from functools import partial

# Sampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Cross Validation
from sklearn.model_selection import cross_val_score

# Pipeline
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_score, recall_score, f1_score

class LabelsPrediction:
    def __init__(self):
        self.grid_params_init()
        self.hyperopt_params_init()
        self.models_init()
    
    def grid_params_init(self):
        """Initialize param space for grid/random search
        """
        self.sgd_param = {
            'rus__sampling_strategy' : ['auto', 0.5, 0.8],
            'rus__replacement' : ['False', 'True'],
            'sgd__loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'sgd__alpha' : [0.00001, 0.0001, 0.001],
            'sgd__learning_rate' : ['optimal', 'invscaling', 'adaptive'],
            'sgd__eta0' : [0.001],
            'sgd__class_weight' : [None]
        }

        self.lr_param = {
            'rus__sampling_strategy' : ['auto', 0.5, 0.8],
            'rus__replacement' : ['False', 'True'],
            'lr__C' : [1,10,100,500,1000],
            'lr__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'lr__class_weight' : [None]
        }

        self.svc_param = {
            'rus__sampling_strategy' : ['auto', 0.5, 0.8],
            'rus__replacement' : ['False', 'True'],
            'svc__C' : [1,10,100,500,1000],
            'svc__class_weight' : [None]
        }

        self.ab_param = {
            'rus__sampling_strategy' : ['auto', 0.5, 0.8],
            'rus__replacement' : ['False', 'True'],
            'ab__n_estimators' : [50, 100, 200, 500, 1000],
            'ab__learning_rate' : [0.001, 0.01, 0.1, 0.5, 1]
        }

        self.bag_param = {
            'rus__sampling_strategy' : ['auto', 0.5, 0.8],
            'rus__replacement' : ['False', 'True'],
            'bag__n_estimators' : [50, 100, 200, 500, 1000],
            'bag__max_samples' : [1, 0.8],
            'bag__max_features' : [1, 0.8, 0.6],
            'bag__bootstrap' : [True, False],
            'bag__bootstrap_features' : [True, False],
            'bag__oob_score' : [False]
        }

        self.rf_param = {
            'rus__sampling_strategy' : ['auto', 0.5, 0.8],
            'rus__replacement' : ['False', 'True'],
            'rf__n_estimators' : [100, 200, 500, 1000],
            'rf__criterion' : ['gini', 'entropy'],
            'rf__max_depth' : [2, 5, 8],
            'rf__min_samples_split' : [5, 10, 20],
            'rf__min_samples_leaf' : [2, 4, 6],
            'rf__max_features' : ['auto', 'sqrt', 'log2', None],
            'rf__bootstrap' : [True, False],
        }

        self.gbc_param = {
            'rus__sampling_strategy' : ['auto', 0.5, 0.8],
            'rus__replacement' : ['False', 'True'],
            'gbc__max_depth' : [2, 5, 8],
            'gbc__max_features' : ['auto', 'sqrt', 'log2', None],
            'gbc__min_samples_leaf' : [2, 4, 6],
            'gbc__min_samples_split': [5, 10, 20],  
            'gbc__n_estimators' : [100, 200, 500, 1000],
            'gbc__learning_rate' : [0.01, 0.1, 0.2],
            'gbc__subsample' : [1, 0.8, 0.6],
        }
                 
    def hyperopt_params_init(self):
        """Initialize param space for hyperopt search
        """
        self.sgd_space = {
            'rus__sampling_strategy' : hp.uniform('rus__sampling_strategy', 0.1, 1),
            'rus__replacement' : hp.choice('rus__replacement', ['False', 'True']),
            'sgd__loss' : hp.choice('sgd__loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),
            'sgd__alpha' : hp.loguniform('sgd__alpha', -5, -3),
            'sgd__learning_rate' : hp.choice('sgd__learning_rate', ['optimal', 'invscaling', 'adaptive']),
            'sgd__eta0' : hp.choice('sgd__eta0', [0.001]),
            'sgd__class_weight' : hp.choice('sgd__class_weight', [None])
        }

        self.lr_space = {
            'rus__sampling_strategy' : hp.uniform('rus__sampling_strategy', 0.1, 1),
            'rus__replacement' : hp.choice('rus__replacement', ['False', 'True']),
            'lr__C' : hp.loguniform('lr__C', 0, 3),
            'lr__solver' : hp.choice('lr__solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            'lr__class_weight' : hp.choice('lr__class_weight', [None])
        }

        self.svc_space = {
            'rus__sampling_strategy' : hp.uniform('rus__sampling_strategy', 0.1, 1),
            'rus__replacement' : hp.choice('rus__replacement', ['False', 'True']),
            'svc__C' : hp.loguniform('svc__C', 0, 3),
            'svc__class_weight' : hp.choice('svc__class_weight', [None])
        }

        self.ab_space = {
            'rus__sampling_strategy' : hp.uniform('rus__sampling_strategy', 0.1, 1),
            'rus__replacement' : hp.choice('rus__replacement', ['False', 'True']),
            'ab__n_estimators' : scope.int(hp.quniform('ab__n_estimators', 10, 1000, 5)),
            'ab__learning_rate' : hp.loguniform('ab__learning_rate', -4, 0)
        }

        self.bag_space = {
            'rus__sampling_strategy' : hp.uniform('rus__sampling_strategy', 0.1, 1),
            'rus__replacement' : hp.choice('rus__replacement', ['False', 'True']),
            'bag__n_estimators' : scope.int(hp.quniform('bag__n_estimators', 10, 1000, 5)),
            'bag__max_samples' : hp.uniform('bag__max_samples', 0.1, 1.0),
            'bag__max_features' :  hp.uniform('bag__max_features', 0.1, 1.0),
            'bag__bootstrap' : hp.choice('bag__bootstrap', [True, False]),
            'bag__bootstrap_features' : hp.choice('bag__bootstrap_features', [True, False]),
            'bag__oob_score' : hp.choice('bag__oob_score', [False])
        }

        self.rf_space = {
            'rus__sampling_strategy' : hp.uniform('rus__sampling_strategy', 0.1, 1),
            'rus__replacement' : hp.choice('rus__replacement', ['False', 'True']),
            'rf__n_estimators' : scope.int(hp.quniform('rf__n_estimators', 10, 1000, 5)),
            'rf__criterion' : hp.choice('rf__criterion', ['gini', 'entropy']),
            'rf__max_depth' : scope.int(hp.quniform('rf__max_depth', 1, 10, 2)),
            'rf__min_samples_split' : scope.int(hp.quniform('rf__min_samples_split', 1, 25, 2)),
            'rf__min_samples_leaf' : scope.int(hp.quniform('rf__min_samples_leaf', 1, 25, 2)),
            'rf__max_features' : hp.choice('rf__max_features', ['auto', 'sqrt', 'log2', None]),
            'rf__bootstrap' : hp.choice('rf__bootstrap', [True, False]),
        }

        self.gbc_space = {
            'rus__sampling_strategy' : hp.uniform('rus__sampling_strategy', 0.1, 1),
            'rus__replacement' : hp.choice('rus__replacement', ['False', 'True']),
            'gbc__max_depth' : scope.int(hp.quniform('gbc__max_depth', 1, 10, 2)),
            'gbc__max_features' : hp.choice('gbc__max_features', ['auto', 'sqrt', 'log2', None]),
            'gbc__min_samples_leaf' : scope.int(hp.quniform('gbc__min_samples_leaf', 1, 25, 2)),
            'gbc__min_samples_split': scope.int(hp.quniform('gbc__min_samples_split', 1, 25, 2)),  
            'gbc__n_estimators' : scope.int(hp.quniform('gbc__n_estimators', 10, 1000, 5)),
            'gbc__learning_rate' : hp.loguniform('gbc__learning_rate', -4, 0),
            'gbc__subsample' : hp.uniform('gbc__subsample', 0.1, 1.0),
        }
    
    def models_init(self):
        """Initialize individual models and preprocessing
        """
        # Impute missing values
        self.imputer = SimpleImputer(strategy='mean')

        # Scaling
        self.scaler = StandardScaler()

        # Models
        self.sgd_model = SGDClassifier()
        self.lr_model = LogisticRegression()
        self.svc_model = LinearSVC(loss='hinge')
        self.ab_model = AdaBoostClassifier()
        self.bag_model = BaggingClassifier()
        self.rf_model = RandomForestClassifier()
        self.gbc_model = GradientBoostingClassifier()

        # Sampling
        self.rus = RandomUnderSampler()
        self.smt = SMOTE()

        # Cross Validation - none
        # Pipeline - none
        # Metrics - none
        
        # Models
        self.grid_models = {
            'sgd': {'est' : self.sgd_model, 'params': self.sgd_param}, 
            'lr': {'est' : self.lr_model, 'params': self.lr_param}, 
            'svc': {'est' : self.svc_model, 'params': self.svc_param}, 
            'ab': {'est' : self.ab_model, 'params': self.ab_param},
            'bag': {'est' : self.bag_model, 'params': self.bag_param}, 
            'rf': {'est' : self.rf_model, 'params': self.rf_param}, 
            'gbc': {'est' : self.gbc_model, 'params': self.gbc_param}
        }
        
        self.hyperopt_models = {
            'sgd': {'est' : self.sgd_model, 'params': self.sgd_space}, 
            'lr': {'est' : self.lr_model, 'params': self.lr_space}, 
            'svc': {'est' : self.svc_model, 'params': self.svc_space}, 
            'ab': {'est' : self.ab_model, 'params': self.ab_space},
            'bag': {'est' : self.bag_model, 'params': self.bag_space}, 
            'rf': {'est' : self.rf_model, 'params': self.rf_space}, 
            'gbc': {'est' : self.gbc_model, 'params': self.gbc_space}
        }
        
    def models_selector(self, tuning):
        """Return the appropriate model items based on the tuning method selected
        """
        if tuning == 'random' or tuning == 'grid':
            return self.grid_models.items()
        elif tuning == 'hyperopt':
            return self.hyperopt_models.items()
        
    def models_fit(self, model, estimator, X_train, y_train, tuning, trials=None):
        """Fit model with optimized hyperparameters and store/print results
        """
        estimator.fit(X_train, y_train)
        
        if tuning == 'random' or tuning == 'grid':
            self.best_estimators[model] = estimator.best_estimator_
            self.best_f1_scores[model] = estimator.best_score_
        elif tuning == 'hyperopt':
            self.best_estimators[model] = estimator
            self.best_f1_scores[model] = -trials.best_trial['result']['loss']    
               
        print('{} best F1 score: {:.4f}'.format(model, self.best_f1_scores[model])) 
        
    def raw_hyperopt_objective(self, model_pipeline, X_train, y_train, params):
        """Objective function to use in hyperopt. 
        We are choose to minimize the negative f1_score to maintain balance between precision and recall
        """
        model_pipeline.set_params(**params)
        
        # Choose scoring='f1' to find best hyperparameters that maximizes f1_score which balances precision and recall
        f1_score = cross_val_score(model_pipeline, X_train, y_train, scoring='f1', cv=10).mean()

        if f1_score > self.best:
            self.best = f1_score

        # print ('New best: {}, {}\n'.format(self.best, params))
        return {'loss': -f1_score, 'status': STATUS_OK}   
        
    def train_models(self, X_train, y_train, tuning='hyperopt'):
        """Hyperparameter tuning.
        Iterate over each model and find best parameter combination using 'tuning' method and cross validation
        Also finally fits a voting classifier with all the optimized models
        """
        
        valid_tuning = ['random', 'grid', 'hyperopt']
        if tuning not in valid_tuning:
            raise ValueError('train_models: tuning must be one of {}'.format(valid_tuning))
        
        start = datetime.now()

        self.best_estimators = {}
        self.best_f1_scores = {}
        self.scores_all = []
        for model, estimator in self.models_selector(tuning):
            self.best = 0
            model_pipeline = Pipeline([
                ('imputer', self.imputer),
                ('scaling', self.scaler),
                ('rus', self.rus),
                (model, estimator['est'])
                ])
            
            if tuning == 'random':
                search = RandomizedSearchCV(model_pipeline, param_distributions=estimator['params'], scoring='f1', n_iter=100, cv=10, verbose=False, n_jobs=-1, iid=False)
                self.models_fit(model, search, X_train, y_train, tuning)
 
            elif tuning == 'grid':
                search = GridSearchCV(model_pipeline, param_grid=param_dist, scoring='f1', cv=10, verbose=True, n_jobs=-1)
                self.models_fit(model, search, X_train, y_train, tuning)
            
            elif tuning == 'hyperopt':
                hyperopt_objective = partial(self.raw_hyperopt_objective, model_pipeline, X_train, y_train)

                trials = Trials()
                best_params = fmin(fn=hyperopt_objective, space=estimator['params'], algo=tpe.suggest, max_evals=20, trials=trials)
                best_params_actual = space_eval(estimator['params'], trials.argmin)

                model_pipeline.set_params(**best_params_actual)              
                self.models_fit(model, model_pipeline, X_train, y_train, tuning, trials)
       
        # Training a Voting classifier with all above estimators
        self.vot_model = VotingClassifier(estimators=list(self.best_estimators.items()), voting='hard')
        scores = cross_val_score(self.vot_model, X_train, y_train, scoring='f1', cv=10)
        self.vot_model.fit(X_train,y_train)
        self.best_estimators['voting'] = self.vot_model
        print('Voting Classifier F1 score mean: {:.4f}, stddev: {:.4f}'.format(scores.mean(), scores.std()))
        
        end = datetime.now()
        print('Train time: {}'.format(end-start))
                   
    def test_models(self, X_train, y_train, X_test, y_test):
        """Plot the performance metrics of Train and Test datasets across all optimized models
        """
        i=0
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
        for dataset in [[X_train, y_train], [X_test, y_test]]:
            model_metrics = {}
            auc_all, precision_all, recall_all, f1_all, model_names = [], [], [], [], []
            for model, estimator in self.best_estimators.items():
                y_pred = self.best_estimators[model].predict(dataset[0])
                fpr, tpr, thresholds = roc_curve(dataset[1], y_pred)
                precision = precision_score(dataset[1], y_pred)
                recall = recall_score(dataset[1], y_pred)
                fmeasure = f1_score(dataset[1], y_pred)
                model_metrics[model] = {}
                model_metrics[model]['auc'] = auc(fpr, tpr)
                model_metrics[model]['precision'] = precision
                model_metrics[model]['recall'] = recall
                model_metrics[model]['f1_score'] = fmeasure

            df = pd.DataFrame(model_metrics)
            df.plot.bar(ax=axes[i], legend=None)
            axes[i].tick_params(direction='out', labelsize=14)
            if i:
                axes[i].legend(fontsize=14, bbox_to_anchor=(1, 1))
            axes[i].set_ylim(0,1)
            i+=1
        axes[0].set_title("Train Dataset", fontsize=14)
        axes[1].set_title("Test Dataset", fontsize=14)