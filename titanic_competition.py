from datamining_machine import DataMiningMachine
from pathlib import Path
import pandas as pd

def main():
    train_file = Path("data") / "train.csv"
    test_file = Path("data") / "test.csv"
    
    training_data  = pd.read_csv(train_file)
    test_data  = pd.read_csv(test_file)

    titanic = DataMiningMachine(training_data,test_data)
    
    pd.set_option('display.max_rows', 10000)
    
    print("Original Training Data: \n",titanic.training_df.head())
    
    """ #######################################################
        P R E - P R O C E S S   T R A I N I N G 
        #######################################################"""
    #Check null values
    # print("Empty values per column: \n",titanic.training_df.isnull().sum()) # -> Age: 177 , Cabin: 687, Embarked: 2

    #drop column name
    titanic.training_df = titanic.drop_column(titanic.training_df,'Name')
    # mining_titanic.drop_column('Cabin') #TODO DELETE THIS
    
    # #Check outcome class distribution (survived), it shouls be close to 50:50
    # print("\n",titanic.training_df['Survived'].value_counts()) # 549:342 -> no to close to 50:50. MEANING? classifiers wont be optimal?
    
    # #scale numerical data 
    titanic.training_df = titanic.scale_numeric_columns(titanic.training_df, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])

    # #pre process categorical data
    titanic.training_df = titanic.encode_categorical_columns(titanic.training_df,['Sex','Embarked'])

    # # encode string columns to numerical value (for column with a lot of classes)
    titanic.training_df = titanic.encode_string_into_incremental_column(titanic.training_df,['Ticket'])
    titanic.training_df = titanic.encode_string_into_incremental_column(titanic.training_df,['Cabin'])
    
    # Preprocess NaN values
    titanic.training_df = titanic.completing_nan_values(titanic.training_df,["Age"])

    # #Set cross validation sets. Split our data into two dataset, training and testing. Ratio of 80:20 for training and testing respectively
    titanic.set_cross_validation_sets('Survived',0.2)

    print("After Preproccessing Training Data: \n",titanic.training_df.head())
   

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
    titanic.test_df = titanic.drop_column(titanic.test_df,'Name')

    titanic.test_df = titanic.scale_numeric_columns(titanic.test_df, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
    titanic.test_df = titanic.encode_categorical_columns(titanic.test_df,['Sex','Embarked'])
    titanic.test_df = titanic.encode_string_into_incremental_column(titanic.test_df,['Ticket'])
    titanic.test_df = titanic.encode_string_into_incremental_column(titanic.test_df,['Cabin'])
    titanic.test_df = titanic.completing_nan_values(titanic.test_df,["Age"])
    titanic.test_df = titanic.completing_nan_values(titanic.test_df,["Fare"])
    
    # titanic.test_df.dropna(subset = ["Fare"], inplace=True) # eliminates 1 row with empty Fare
    titanic.test_df['Embarked_nan'] = 0.0 #adds a column Embarked_nan to match training data. In the training df, Embarked_nan its created when encoding Embarked column
    titanic.test_df['Survived'] = 0 # only to being able to re ordering column. will drop this column later
    
    # print("\n\n",titanic.training_df.head())
    # print("\n\n",titanic.test_df.head())

    titanic.test_df = titanic.test_df[titanic.training_df.columns] #re order test columns to be equal to train
    titanic.test_df = titanic.drop_column(titanic.test_df,'Survived') #drop artificial column
   
    # print("\n\n",titanic.test_df.head())

    """ #######################################################
        M A K E   P R E D I C T I O N S
        #######################################################"""

    #create prediction
    dec_tree_predict = titanic.predict(dec_tree_model)
    # prediction = pd.DataFrame(dec_tree_predict, columns=['Survived']).astype(int)

    rand_forest_predict = titanic.predict(dec_tree_model)
    # prediction = pd.DataFrame(rand_forest_predict, columns=['Survived']).astype(int)

    naive_bayes_predict = titanic.predict(naive_bay_model)
    prediction = pd.DataFrame(naive_bayes_predict, columns=['Survived']).astype(int)    

    passengerID_column = titanic.test_df['PassengerId']

    concat = prediction.merge(passengerID_column.to_frame(), left_index=True, right_index=True)
    # print("\n\nRANDOM FOREST PREDICTION: \n",titanic.predict(rand_forest_model),"\n")
    # print("\n\nNAIVE BAYES PREDICTION: \n",titanic.predict(naive_bay_model),"\n")
    # print("\n\nK_NEAREST PREDICTION: \n",titanic.predict(k_nearest_model),"\n")

    out_file = Path("data") / "out_predict.csv"
    concat.to_csv(out_file, index=False)    

if __name__ == "__main__":
    main()