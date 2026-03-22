# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl    
from sklearn.model_selection import train_test_split
np.bool = np.bool_
import platform

if platform.system()== 'Windows':
    pass
else:
    pl.rc('font', family='serif',  serif='Times')
    pl.rc('text', usetex=True)
    #pl.rc('font',**{'family':'serif','serif':['Palatino']})

from sklearn.preprocessing import LabelEncoder

dataset='RE'
target_column='Removal efficiency %'
test_size=.8
seed=1
plot=True
#%%
def read_data_pfoa(target='Removal efficiency %', dataset=None,  test_size=None, seed=None, type_dataset='orginal', plot=False):
    #%% 
    fn='./data/pfoa-data.csv'
    data = pd.read_csv(fn)
    # Step 2: Preprocess the data
    # Check for missing values and drop rows with missing values in the target column
    target_column = 'Removal efficiency %'
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    label_encoders = {}
    for col in data.select_dtypes(include=['object', 'category']).columns:
        if col != target_column:  # Evitar codificar a variável resposta se for categórica
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le  # Guarda o encoder para uso posterior (ex: inversão)


    data = data.dropna(subset=[target_column])

    cols =[
        #'Anode used', 'Cathode used', 
        'spacing(cm)', 
        #'Electrolyte used',
           'electrolyte concentration (g/L)', 'initial pH', 'Temperature',
           'initial con. mg/L', 'current density (mA/cm2)',
           'electrolysis time (minutes)', 
           #'water matrix', 
           'Removal efficiency %',
           ] 
    if type_dataset == 'expanded':
        cols =[
           'Anode used', 'Cathode used', 
           'spacing(cm)', 
           'Electrolyte used',
              'electrolyte concentration (g/L)', 'initial pH', 'Temperature',
              'initial con. mg/L', 'current density (mA/cm2)',
              'electrolysis time (minutes)', 
              'water matrix', 
              'Removal efficiency %',
              ]  
        
    data=data[cols]
    # Define features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    #X.columns = ['x_'+('%2.2d'%(i+1)) for i in range(X.shape[1])]
    X.columns = ['$x_{'+str(i+1)+'}$' for i in range(X.shape[1])]
    if type_dataset == 'expanded':
        X.columns = ['$x_{'+str(i+0)+'}$' for i in range(X.shape[1])]
        
    y.columns=['y']
    
    target_names=[target_column]
    # Ensure all features are numeric (convert categorical variables to numeric if necessary)
    X = pd.get_dummies(X, drop_first=True)

    variable_names=feature_names = X.columns.tolist()
    n_features = len(feature_names)    
    target_names=['$y$']
    #variable_names=list(data.columns.drop(target_names))
    #variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    #X=data[variable_names+target_names]
    X[target_names[0]]=y.values
    X.dropna(inplace=True)
       
    categorical_columns=[]   
    for cc in categorical_columns:
        #print(cc)       
        le = LabelEncoder(); 
        le.fit(X[cc].values.ravel()); 
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1,1)
        #classes = dict(zip(le.transform(le.classes_), le.classes_))
        
    if test_size==0 or test_size==None:    
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test , y_test  = pd.DataFrame([[],]).values, pd.DataFrame([[],]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed)
        y_train, y_test = y_train.ravel(), y_test.ravel()

    
    df = X[variable_names+target_names].copy()
    #df.columns = [x.replace('(wt%)','') for x in df.columns]
    
    if plot:
        pl.figure(figsize=(5, 4))
        corr = df.corr().round(2)
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
        #heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
        heatmap.set_title(dataset+': Correlation Heatmap ', fontdict={'fontsize':12}, pad=12);
        pl.savefig(dataset+'_heatmap_correlation'+'.png',  bbox_inches='tight', dpi=300)
        pl.show()
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # def corrdot(*args, **kwargs):
        #     corr_r = args[0].corr(args[1], 'pearson')
        #     corr_text = f"{corr_r:2.2f}".replace("0.", ".")
        #     ax = plt.gca()
        #     ax.set_axis_off()
        #     marker_size = abs(corr_r) * 10000
        #     ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
        #                vmin=-1, vmax=1, transform=ax.transAxes)
        #     font_size = abs(corr_r) * 40 + 5
        #     ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
        #                 ha='center', va='center', fontsize=font_size)
        
        # sns.set(style='white', font_scale=1.6)
        # iris = df
        # g = sns.PairGrid(iris, aspect=1.4, diag_sharey=False)
        # g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
        # g.map_diag(sns.distplot, kde_kws={'color': 'black'})
        # g.map_upper(corrdot)
    
    n=len(y_train);     
    n_samples, n_features = X_train.shape 

    df_train=pd.DataFrame(X_train, columns=variable_names); df_train[target_names]=y_train.reshape(-1,1)
    stat_train = df_train.describe().T
    #print(stat_train.to_latex(),)
    stat_train.to_latex(buf=(dataset+'_train'+'.tex').lower(), float_format="%.2f", index=True, caption='Basic statistics for dataset '+dataset+'.')

    
    task = 'regression'# if target_names[0]=='UCS' else 'classification'
    regression_data =  {
      'task'            : task,
      'name'            : dataset,
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'y_train'         : y_train.reshape(1,-1),
      'X_test'          : X_test,
      'y_test'          : y_test.reshape(1,-1),
      'targets'         : target_names,
       #'true_labels'     : classes,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'reference'       : "https://doi.org/10.1016/j.jenvman.2024.122857",
      'items'           : None,
      'normalize'       : None,
      }
    #%%
    return regression_data
        
