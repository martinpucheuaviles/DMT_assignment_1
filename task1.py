from datamining_machine import DataMiningMachine
from pathlib import Path
import pandas as pd

def main():
    print()
    task_1b()

def task_1b():
    #TASK 1B
    data_folder = Path("data")
    file_to_open = data_folder / "bank.csv"
    data  = pd.read_csv(file_to_open)
    
    mining_bank = DataMiningMachine(data)
    print(mining_bank.training_df.head(),"\n")

    # # Drop 'duration' column (explanation on the tutorial)
    mining_bank.drop_column('duration')

    #Scale numerical data to avoid outlier presence that can significantly affect our model
    mining_bank.scale_numeric_columns(['age', 'balance', 'day', 'campaign', 'pdays', 'previous'])

    #pre-process our categorical data from words to number to make it easier for the computer to understands
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
