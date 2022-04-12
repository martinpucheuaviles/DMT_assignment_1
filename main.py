
# from email import header
# from this import d
# from unicodedata import numeric
# from wsgiref import headers
from jinja2 import Undefined
import pandas as pd
from tabulate import tabulate
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics,tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class DataMiningMachine:
    def __init__(self,data):

        self.df = data
        self.X_train = None
        self.X_test  = None 
        self.y_train = None
        self.y_test  = None

        self.decision_tree_classifier = None
        self.random_forest_classifier = None
        self.naive_bayes_classifier = None
        self.k_nearest_neighbors_classifier = None
    
    def drop_column(self,column):
        self.df = self.df.drop(column, axis=1)

    def scale_numeric_columns(self,num_cols):
        """
        Scale our numerical data to avoid outlier presence that can significantly affect our model
        Args:
            @num_cols:  (list of column indexes) numerical columns to scale
        """

        scaler = StandardScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
    
    def encode_categorical_columns(self,cat_cols):
        """
        Pre-process categorical data from words to number
        Args:
            @cat_cols:  (list of column indexes) categorical columns to scale
        """        

        encoder = OneHotEncoder(sparse=False)

        # Encode Categorical Data
        df_encoded = pd.DataFrame(encoder.fit_transform(self.df[cat_cols]))
        df_encoded.columns = encoder.get_feature_names(cat_cols)


        # Replace Categotical Data with Encoded Data
        self.df = self.df.drop(cat_cols ,axis=1)
        self.df = pd.concat([df_encoded, self.df], axis=1)        

    def encode_categorical_outcome(self,out_col):
        """
        Pre-process categorical outcome column from words to numbers
        Args:
            @out_col:  (string) name of column
        """   
        self.df[out_col] = self.df[out_col].apply(lambda x: 1 if x == 'yes' else 0)        

    def set_cross_validation_sets(self,target_col,test_size):
        # Select Features
        feature = self.df.drop(target_col, axis=1)

        # Select Target
        target = self.df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(feature , target, 
                                                            shuffle = True, 
                                                            test_size=0.2, 
                                                            random_state=1)


        # Show the Training and Testing Data
        # print('Shape of training feature:', self.X_train.shape)
        # print('Shape of testing feature:', self.X_test.shape)
        # print('Shape of training label:', self.y_train.shape)
        # print('Shape of training label:', self.y_test.shape)

    def evaluate(self,model, x_test, y_test):
        from sklearn import metrics

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

    def build_naive_bayes_model(self):
        self.naive_bayes_classifier = GaussianNB()
        self.naive_bayes_classifier.fit(self.X_train, self.y_train)                

        """GaussianNB(priors=None, var_smoothing=1e-09)"""

    def build_k_nearest_neighbors_model(self):
        self.k_nearest_neighbors_classifier = KNeighborsClassifier()
        self.k_nearest_neighbors_classifier.fit(self.X_train, self.y_train)        

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
            model_evals[model] = self.evaluate_model(model)

        print(tabulate(
            [[key,value['acc'],value['prec'],value['rec'],value['f1'],value['kappa'],value['auc'],value['cm']] for key,value in model_evals.items()],
             headers=["Model","Accuracy","Precision","Recall","F1 Score","Cohens Kappa Score","Area Under Curve","Confusion Matrix"]
        ))


    def preprocess_numerical_column(self,column):

        data_ = self.df
        data_.iloc[:,column] = pd.to_numeric(data_.iloc[:,column], errors='coerce',downcast='integer')

        self.df = data_
            
    def get_number_of_records(self):
        return len(self.df)

    def get_numerical_column(self,column):
        
        column = self.df.iloc[:,column]
        column = pd.to_numeric(column, errors='coerce',downcast='integer')

        frame = {column.name: column}
        
        new_df = pd.DataFrame(frame)
        
        return new_df

    def get_min_from_column(self,column):
        
        data_ = self.df.iloc[:,column]
        data_ = pd.to_numeric(data_, errors='coerce',downcast='integer')
        
        column_min = data_.min()

        return column_min

    def get_max_from_column(self,column):
        
        data_ = self.df.iloc[:,column]
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
            
            data_ = self.df.iloc[:,[attribute,classify_on]] #duplicate dataframe with only two columns: current and predicted attrs
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

        print(tabulate([[self.df.columns[attribute],general_rules[attribute],attr_total_errors[attribute]] for attribute in attributes], headers=["Attribute","Rules","Total Errors"]))

def main_1b():

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
   
def main():
    data_folder = Path("data")
    file_to_open = data_folder / "bank.csv"
    data  = pd.read_csv(file_to_open)
    
    mining_bank = DataMiningMachine(data)
    print(mining_bank.df.head(),"\n")

    # # Drop 'duration' column (explanation on the tutorial)
    mining_bank.drop_column('duration')

    #Scale numerical data to avoid outlier presence that can significantly affect our model
    mining_bank.scale_numeric_columns(['age', 'balance', 'day', 'campaign', 'pdays', 'previous'])

    #Scale our numerical data to avoid outlier presence that can significantly affect our model
    mining_bank.encode_categorical_columns(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])

    #Scale categorical outcome
    mining_bank.encode_categorical_outcome('deposit')

    #Set cross validation sets. Split our data into two dataset, training and testing. Ratio of 80:20 for training and testing respectively
    mining_bank.set_cross_validation_sets('deposit',0.2)
    
    #build and test decision tree classifier
    mining_bank.build_decision_tree_model()
    # mining_bank.evaluate_model(mining_bank.decision_tree_classifier)

    #build and test random forest classifier
    mining_bank.build_random_forest_model()
    # mining_bank.evaluate_model(mining_bank.random_forest_classifier)    

    #build and test naive bayes classifier
    mining_bank.build_naive_bayes_model()
    # mining_bank.evaluate_model(mining_bank.naive_bayes_classifier)

    #build and test k-nearest neighbors classifier
    mining_bank.build_k_nearest_neighbors_model()
    # mining_bank.evaluate_model(mining_bank.k_nearest_neighbors_classifier)

    mining_bank.print_metrics()

    
    # print(mining_bank.df)






    




if __name__ == "__main__":
    main()

    