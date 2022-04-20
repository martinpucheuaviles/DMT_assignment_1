from datamining_machine import DataMiningMachine
from pathlib import Path
import pandas as pd

def main():
    train_file = Path("data") / "train.csv"
    test_file = Path("data") / "test.csv"
    
    training_data  = pd.read_csv(train_file)
    test_data  = pd.read_csv(test_file)

    titanic = DataMiningMachine(training_data,test_data)
    
    # pd.set_option('display.max_rows', 10000)
    
    print("Original Training Data: \n",titanic.training_df.head())
    
    """ #######################################################
        P R E - P R O C E S S   T R A I N I N G 
        #######################################################"""
    #Check null values
    # print("Empty values per column: \n",titanic.training_df.isnull().sum()) # -> Age: 177 , Cabin: 687, Embarked: 2

    #drop column name
    # titanic.training_df = titanic.drop_column(titanic.training_df,'Name')
    # mining_titanic.drop_column('Cabin') #TODO DELETE THIS
    
    # #Check outcome class distribution (survived), it shouls be close to 50:50
    # print("\n",titanic.training_df['Survived'].value_counts()) # 549:342 -> no to close to 50:50. MEANING? classifiers wont be optimal?
    
    # passengerId_column = titanic.training_df['PassengerId']
    # titanic.training_df = titanic.drop_column(titanic.training_df,'PassengerId')
    
    print("Original Training Data: \n",titanic.training_df.head())
    
    #Combine two columns into one: Family_size  = SibSp + Parch
    titanic.sum_two_columns(titanic.training_df,new_col_name="Family Size",first_col="SibSp",second_col="Parch")

    # # encode string columns to numerical value (for column with a lot of classes)
    titanic.training_df = titanic.encode_string_into_incremental_column(titanic.training_df,['Name'])
    titanic.training_df = titanic.encode_string_into_incremental_column(titanic.training_df,['Ticket'])
    titanic.training_df = titanic.encode_string_into_incremental_column(titanic.training_df,['Cabin'])

    # #scale numerical data 
    # titanic.training_df = titanic.scale_numeric_columns(titanic.training_df, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
    # titanic.training_df = titanic.scale_numeric_columns(titanic.training_df, ['Pclass', 'Age', 'SibSp', 'Parch'])
    titanic.training_df = titanic.scale_numeric_columns(titanic.training_df, ['Age','Fare','Family Size','Name','Ticket','Cabin','SibSp','Parch'])

    # titanic.quantile_discretization_columns(titanic.training_df,num_quantiles=4,col="Fare",labels=["4th","3rd","2nd","1st"])

    # #pre process categorical data
    print("Empty values per column: \n",titanic.training_df.isnull().sum()) # -> Age: 177 , Cabin: 687, Embarked: 2
    # titanic.training_df = titanic.encode_categorical_columns(titanic.training_df,['Sex','Embarked','Fare'])
    titanic.training_df = titanic.encode_categorical_columns(titanic.training_df,['Sex','Embarked'])
    

    # Preprocess NaN values
    titanic.training_df = titanic.completing_nan_values(titanic.training_df,["Age"])

    # #Set cross validation sets. Split our data into two dataset, training and testing. Ratio of 80:20 for training and testing respectively
    
    titanic.set_cross_validation_sets('Survived',0.15) # 0.15 optimal until now

    print("\n\nAfter Preproccessing Training Data: \n",titanic.training_df.head())
   

    """ #######################################################
         C R E A T E   A N D   E V A L U A T E   M O D E L S 
        #######################################################"""   
    

    # # #build and test decision tree classifier
    dec_tree_model = titanic.build_decision_tree_model()
    # # titanic.evaluate_model(titanic.decision_tree_classifier)

    # # #build and test random forest classifier
    rand_forest_model = titanic.build_random_forest_model()
    # # # titanic.evaluate_model(titanic.random_forest_classifier)    

    # # #build and test naive bayes classifier
    naive_bay_model = titanic.build_naive_bayes_model()
    # # # titanic.evaluate_model(titanic.naive_bayes_classifier)

    # # #build and test k-nearest neighbors classifier
    k_nearest_model = titanic.build_k_nearest_neighbors_model()
    # # # titanic.evaluate_model(titanic.k_nearest_neighbors_classifier)

    print("EVALUATION METRICS FOR TRAINING DATA: ")
    titanic.print_metrics()

    """ #######################################################
        P R E - P R O C E S S   T E S T   D A T A 
        #######################################################"""
    # print("Empty values per column: \n",titanic.test_df.isnull().sum()) # -> Age: 86 , Cabin: 327, Fare 1
    # titanic.test_df = titanic.drop_column(titanic.test_df,'Name')

    print("\n\nTEST: before pre proccess:")
    print(titanic.test_df.head())    

    # passengerId_column = titanic.test_df['PassengerId']
    # titanic.test_df = titanic.drop_column(titanic.test_df,'PassengerId')    

    titanic.test_df = titanic.sum_two_columns(titanic.test_df,new_col_name="Family Size",first_col="SibSp",second_col="Parch")

    titanic.test_df = titanic.encode_string_into_incremental_column(titanic.test_df,['Name'])
    titanic.test_df = titanic.encode_string_into_incremental_column(titanic.test_df,['Ticket'])
    titanic.test_df = titanic.encode_string_into_incremental_column(titanic.test_df,['Cabin'])

    # titanic.test_df = titanic.scale_numeric_columns(titanic.test_df, ['Age','Fare'])
    titanic.test_df = titanic.scale_numeric_columns(titanic.test_df, ['Age','Fare','Family Size','Name','Ticket','Cabin','SibSp','Parch'])
    
    # titanic.test_df = titanic.quantile_discretization_columns(titanic.test_df,num_quantiles=4,col="Fare",labels=["4th","3rd","2nd","1st"])
    titanic.test_df = titanic.encode_categorical_columns(titanic.test_df,['Sex','Embarked'])

    titanic.test_df = titanic.completing_nan_values(titanic.test_df,["Age"])
    
    # titanic.test_df = titanic.completing_nan_values(titanic.test_df,["Fare"])
    
    titanic.test_df['Embarked_nan'] = 0.0 #adds a column Embarked_nan to match training data. In the training df, Embarked_nan its created when encoding Embarked column
    titanic.test_df['Survived'] = 0 # only to being able to re ordering column. will drop this column later
    

    titanic.test_df = titanic.test_df[titanic.training_df.columns] #re order test columns to be equal to train
    titanic.test_df = titanic.drop_column(titanic.test_df,'Survived') #drop artificial column
   
    titanic.test_df = titanic.completing_nan_values(titanic.test_df,["Fare"])

    print("\n\nTEST: after pre proccess:")
    print(titanic.test_df.head())

    print("Empty values per column: \n",titanic.test_df.isnull().sum()) # -> Age: 177 , Cabin: 687, Embarked: 2
    # print("\n\n",titanic.test_df.head())

    """ #######################################################
        M A K E   P R E D I C T I O N S
        #######################################################"""

    #create predictions
    dec_tree_prediction     = pd.DataFrame(titanic.predict(dec_tree_model), columns=['Survived']).astype(int)
    rand_forest_prediction  = pd.DataFrame(titanic.predict(rand_forest_model), columns=['Survived']).astype(int)
    naive_bayes_prediction  = pd.DataFrame(titanic.predict(naive_bay_model), columns=['Survived']).astype(int)
    k_nearest_prediction    = pd.DataFrame(titanic.predict(k_nearest_model), columns=['Survived']).astype(int)
    
    # titanic.test_df['PassengerId'] = passengerId_column
    passengerID_column = titanic.test_df['PassengerId']

    dec_tree_out_df    = dec_tree_prediction.merge(passengerID_column.to_frame(), left_index=True, right_index=True)
    rand_forest_out_df = rand_forest_prediction.merge(passengerID_column.to_frame(), left_index=True, right_index=True)
    naive_bayes_out_df = naive_bayes_prediction.merge(passengerID_column.to_frame(), left_index=True, right_index=True)
    k_nearest_out_df   = k_nearest_prediction.merge(passengerID_column.to_frame(), left_index=True, right_index=True)
    
    prefix = "xv15_normFareAge_yesID_"

    dec_tree_out_file = Path("data") / (prefix+"decision_tree.csv")
    rand_forest_out_file = Path("data") / (prefix+"random_forest.csv")
    naive_bayes_out_file = Path("data") / (prefix+"naive_bayes.csv")
    k_nearest_out_file = Path("data") / (prefix+"k_nearest.csv")
    
    dec_tree_out_df.to_csv(dec_tree_out_file, index=False)    
    rand_forest_out_df.to_csv(rand_forest_out_file, index=False)    
    naive_bayes_out_df.to_csv(naive_bayes_out_file, index=False)    
    k_nearest_out_df.to_csv(k_nearest_out_file, index=False)    

if __name__ == "__main__":
    main()