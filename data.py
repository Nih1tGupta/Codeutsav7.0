import csv

def get_college_data(name):
    with open("EngineeringRanking.csv", "r") as f:
        csv_data = csv.reader(f)
        for line in csv_data:
            if line[1] == name:
                return line
# print(get_college_data("Indian Institute of Technology Madras"))