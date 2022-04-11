
from email import header
from this import d
from unicodedata import numeric
from wsgiref import headers
import pandas as pd
from tabulate import tabulate
from pathlib import Path

class DataMiningMachine:
    def __init__(self,data):

        self.data = data
            
    def get_number_of_records(self):
        return len(self.data)

    def get_numerical_column(self,column):
        
        column = self.data.iloc[:,column]
        column = pd.to_numeric(column, errors='coerce',downcast='integer')

        frame = {column.name: column}
        
        new_df = pd.DataFrame(frame)
        
        return new_df

    def get_min_from_column(self,column):
        
        data_ = self.data.iloc[:,column]
        data_ = pd.to_numeric(data_, errors='coerce',downcast='integer')
        
        column_min = data_.min()

        return column_min

    def get_max_from_column(self,column):
        
        data_ = self.data.iloc[:,column]
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
            
            data_ = self.data.iloc[:,[attribute,classify_on]] #duplicate dataframe with only two columns: current and predicted attrs
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
                
                # print("Data filtered for {} = {} \n {}\n".format(column_name,attr_value,data__))
                # print("Count of predicted class:\n{}\n".format(data__.iloc[:,1].value_counts()))
                # print("^^The most frequent class for attribute [{}] = [{}] is: [{}]".format(cpolumn_name,attr_value,most_frequent_class))
                # print("errors=",len(data_errors))
                # print("\n\n")

            attr_total_errors[attribute] = error_count

        print(tabulate([[self.data.columns[attribute],general_rules[attribute],attr_total_errors[attribute]] for attribute in attributes], headers=["Attribute","Rules","Total Errors"]))

def main():

    data_folder = Path("data")

    file_to_open = data_folder / "ODI-2022.csv"
    data  = pd.read_csv(file_to_open,sep=";")
    
    pd.set_option('display.max_rows', data.shape[0]+1)
    
    mining_machine = DataMiningMachine(data)

    print("\n\n#################################################################\n\n")
    # print("The number of records is:", mining_machine.get_number_of_records())
    # column = 11; print("The range of values in column {} is: [{} , {}]".format(column,mining_machine.get_min_from_column(column),mining_machine.get_max_from_column(column)))
    # print("NEW DF:\n",mining_machine.get_numerical_column(column))

    mining_machine.classification_1R([2,3,4,6],5) 


if __name__ == "__main__":
    main()