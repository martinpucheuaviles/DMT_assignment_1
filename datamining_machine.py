from cProfile import label
from jinja2 import Undefined
import pandas as pd
from tabulate import tabulate
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics,tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class DataMiningMachine:
    def __init__(self,training_data,test_data=None):

        self.training_df = training_data
        self.test_df = test_data
        self.X_train = None
        self.X_test  = None 
        self.y_train = None
        self.y_test  = None

        self.decision_tree_classifier = None
        self.random_forest_classifier = None
        self.naive_bayes_classifier = None
        self.k_nearest_neighbors_classifier = None
    
    def drop_column(self,df,column):
        
        df = df.drop(column, axis=1)
        
        return df

    def scale_numeric_columns(self,df,num_cols):
        """
        Scale our numerical data to avoid outlier presence that can significantly affect our model
        Args:
            @num_cols:  (list of column indexes) numerical columns to scale
        """

        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        return df
    
    def encode_categorical_columns(self,df,cat_cols):
        """
        Pre-process categorical data from words to number
        Args:
            @cat_cols:  (list of column indexes) categorical columns to scale
        """        

        encoder = OneHotEncoder(sparse=False)

        # Encode Categorical Data
        df_encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]))
        df_encoded.columns = encoder.get_feature_names(cat_cols)


        # Replace Categotical Data with Encoded Data
        df = df.drop(cat_cols ,axis=1)
        df = pd.concat([df_encoded, df], axis=1)        

        return df

    def encode_categorical_outcome(self,out_col):
        """
        Pre-process categorical outcome column from words to numbers
        Args:
            @out_col:  (string) name of column
        """   
        self.training_df[out_col] = self.training_df[out_col].apply(lambda x: 1 if x == 'yes' else 0)        

    def convert_column_to_only_numerical(self,col):
        """
        Convert all values from columns to only numerical part (e.g. 'ABC 123' -> '123')
        """   
        self.training_df[col] = self.training_df[col].apply(lambda x: "".join(filter(str.isdigit, x)))

    def encode_string_into_incremental_column(self,df,string_cols):
        """
        Use LabelEncoder to encode target labels with value between 0 and n_classes-1.
        Args:
            @string_cols:  (list of column indexes) string columns to transform
        """        
        encoder = LabelEncoder()

        # Encode Categorical Data
        df_encoded = pd.DataFrame(df[string_cols].apply(encoder.fit_transform))
        df_encoded.columns = df_encoded.columns

        # Replace Categotical Data with Encoded Data
        df = df.drop(string_cols ,axis=1)
        df = pd.concat([df_encoded, df], axis=1)     

        return df

    def quantile_discretization_columns(self,df,col,num_quantiles,labels):
        """
        Quantile-based discretization of continous numeric column
        """

        df[col]  = pd.qcut(df[col],q=num_quantiles,labels=labels)

        return df

    def completing_nan_values(self,df,cols):
        """
        Completing missing numerical values (NaN) from columns. NaN values are replaced by the column's mean
        Args:
            @string_cols:  (list of column indexes) string columns to complete

        """
        # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
        
        imp_mean = imp_mean.fit(df[cols])

        df[cols] = imp_mean.transform(df[cols])

        return df

    def sum_two_columns(self,df,new_col_name,first_col,second_col):
        
        df.apply(lambda row: row[first_col] + row[second_col], axis=1)
        df[new_col_name] = df.apply(lambda row: row[first_col] + row[second_col], axis=1)

        return df
    def set_cross_validation_sets(self,target_col,size):
        # Select Features
        feature = self.training_df.drop(target_col, axis=1)

        # Select Target
        target = self.training_df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(feature , target, 
                                                            shuffle = True, 
                                                            test_size=size, 
                                                            random_state=1)

        # Show the Training and Testing Data
        # print('Shape of training feature:', self.X_train.shape)
        # print('Shape of testing feature:', self.X_test.shape)
        # print('Shape of training label:', self.y_train.shape)
        # print('Shape of training label:', self.y_test.shape)

    def build_decision_tree_model(self):
        # Building Decision Tree model 
        self.decision_tree_classifier = tree.DecisionTreeClassifier(random_state=0)
        self.decision_tree_classifier.fit(self.X_train, self.y_train)        

        """DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='best')"""

        return self.decision_tree_classifier

    def build_random_forest_model(self):
        # Building Random Forest model 
        self.random_forest_classifier = RandomForestClassifier(random_state=0)
        self.random_forest_classifier.fit(self.X_train, self.y_train)

        """RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)"""

        return self.random_forest_classifier

    def build_naive_bayes_model(self):
        self.naive_bayes_classifier = GaussianNB()
        self.naive_bayes_classifier.fit(self.X_train, self.y_train)                

        """GaussianNB(priors=None, var_smoothing=1e-09)"""

        return self.naive_bayes_classifier

    def build_k_nearest_neighbors_model(self):
        self.k_nearest_neighbors_classifier = KNeighborsClassifier()
        self.k_nearest_neighbors_classifier.fit(self.X_train, self.y_train)        

        return self.k_nearest_neighbors_classifier

    def evaluate(self,model, x_test, y_test):

        # Predict Test Data 
        y_pred = model.predict(x_test)

        # Calculate accuracy, precision, recall, f1-score, and kappa score
        acc = metrics.accuracy_score(y_test, y_pred)
        prec = metrics.precision_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        kappa = metrics.cohen_kappa_score(y_test, y_pred)

        # Calculate area under curve (AUC)
        y_pred_proba = model.predict_proba(x_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        # Display confussion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)

        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
                'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}

    def predict(self,model):
        # Predict Test Data 
        y_pred = model.predict(self.test_df)

        return y_pred

    def evaluate_model(self,model):

        evaluation = self.evaluate(model, self.X_test, self.y_test)

        # Print result
        # print('Accuracy:', evaluation['acc'])
        # print('Precision:', evaluation['prec'])
        # print('Recall:', evaluation['rec'])
        # print('F1 Score:', evaluation['f1'])
        # print('Cohens Kappa Score:', evaluation['kappa'])
        # print('Area Under Curve:', evaluation['auc'])
        # print('Confusion Matrix:\n', evaluation['cm'])

        return evaluation

    def print_metrics(self):

        models = [self.decision_tree_classifier,
                    self.random_forest_classifier,
                    self.naive_bayes_classifier,
                    self.k_nearest_neighbors_classifier]

        model_evals = {}
        for model in models:
            if model is not None:
                model_evals[model] = self.evaluate_model(model)

        print(tabulate(
            [[key,value['acc'],value['prec'],value['rec'],value['f1'],value['kappa'],value['auc'],value['cm']] for key,value in model_evals.items()],
             headers=["Model","Accuracy","Precision","Recall","F1 Score","Cohens Kappa Score","Area Under Curve","Confusion Matrix"]
        ))

    def preprocess_numerical_column(self,column):

        data_ = self.training_df
        data_.iloc[:,column] = pd.to_numeric(data_.iloc[:,column], errors='coerce',downcast='integer')

        self.training_df = data_
            
    def get_number_of_records(self):
        return len(self.training_df)

    def get_numerical_column(self,column):
        
        column = self.training_df.iloc[:,column]
        column = pd.to_numeric(column, errors='coerce',downcast='integer')

        frame = {column.name: column}
        
        new_df = pd.DataFrame(frame)
        
        return new_df

    def get_min_from_column(self,column):
        
        data_ = self.training_df.iloc[:,column]
        data_ = pd.to_numeric(data_, errors='coerce',downcast='integer')
        
        column_min = data_.min()

        return column_min

    def get_max_from_column(self,column):
        
        data_ = self.training_df.iloc[:,column]
        data_ = pd.to_numeric(data_, errors='coerce',downcast='integer')
        
        column_max = data_.max()

        return column_max

    def classification_1R(self,attributes,classify_on):
        """
        1R Classifier Algorithm
        
        Args:
            @attributes:  (list of column indexes) columns we will use to classify e.g. weather 
            @classify_on: (column index)           column on which we want to classify - response columne.g. play? yes/no
        
        Returns:
        """    
        general_rules = {}
        attr_total_errors = {}    
        
        for attribute in attributes:
            attribute_rules = {}
            general_rules[attribute] = attribute_rules
            
            data_ = self.training_df.iloc[:,[attribute,classify_on]] #duplicate dataframe with only two columns: current and predicted attrs
            column_name = data_.columns[0] #only for printing motives
            attr_values = data_.iloc[:,0].unique(); #same as data_[column_name].unique();

            error_count = 0
            for attr_value in attr_values:
                
                data__ = data_.loc[data_[column_name] == attr_value] #get data filtered for attr value
                
                most_frequent_class = data__.iloc[:,1].value_counts().idxmax() #get the most frequent answer in output class for current attr value
                
                attribute_rules[attr_value] = most_frequent_class #create rule

                #now get the errors:
                data_errors = data__.loc[data__.iloc[:,1] != most_frequent_class] #dataset with the rows where the attribute value dont match the most frequent class of predicted attribute
                
                error_count += len(data_errors)
                
            attr_total_errors[attribute] = error_count

        print(tabulate([[self.training_df.columns[attribute],general_rules[attribute],attr_total_errors[attribute]] for attribute in attributes], headers=["Attribute","Rules","Total Errors"]))

def main_1a():

    data_folder = Path("data")

    file_to_open = data_folder / "ODI-2022.csv"
    data  = pd.read_csv(file_to_open,sep=";")
    
    pd.set_option('display.max_rows', data.shape[0]+1)
    
    mining_machine = DataMiningMachine(data)

    print("\n\n#################################################################\n\n")
    # print("The number of records is:", mining_machine.get_number_of_records())
    # column = 11; print("The range of values in column {} is: [{} , {}]".format(column,mining_machine.get_min_from_column(column),mining_machine.get_max_from_column(column)))
    # print("NEW DF:\n",mining_machine.get_numerical_column(column))
    # mining_machine.preprocess_numerical_column(11) # preprocess numerical column
    # mining_machine.classification_1R([2,3,4,6],5) 
   


    