
from this import d
from unicodedata import numeric
import pandas as pd


class DataMiningMachine:
    def __init__(self,data):

        self.data = data
            
    def get_number_of_records(self):
        return len(self.data)

    def get_numerical_column(self,column):
        
        column = self.data.iloc[:,column]
        column = pd.to_numeric(column, errors='coerce',downcast='integer')



        frame = { column.name:column}
        
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


def main():
    data  = pd.read_csv('data\ODI-2022.csv',sep=";")
    # data  = pd.read_excel('data\ODI-2022.xlsx')
    pd.set_option('display.max_rows', data.shape[0]+1)
    
    mining_machine = DataMiningMachine(data)

    print("#################################################################")
    # print("The number of records is:", mining_machine.get_number_of_records())
    column = 11; print("The range of values in column {} is: [{} , {}]".format(column,mining_machine.get_min_from_column(column),mining_machine.get_max_from_column(column)))

    print("NEW DF:\n",mining_machine.get_numerical_column(column))
    
    
    # print(data.iloc[:,12])


if __name__ == "__main__":
    main()