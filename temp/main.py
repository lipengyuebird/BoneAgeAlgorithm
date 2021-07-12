# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import xlrd
import csv
import os


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('BoneAgeAlgorithm')

    img_list = []
    multiple_dict = {}

    with open('C:\\BoneAgeAssessment\\Route1\\dataset\\annos\\special_data.csv', 'r') as f:

        for line in csv.reader(f):
            try:
                multiple_dict[line[0]] = int(line[3])
            except ValueError:
                pass

    print(multiple_dict)

    with open('C:\\BoneAgeAssessment\\Route2\\dataset\\annos\\score.csv', 'r', encoding='utf-8') as f:

        title = f.readline()
        reader = csv.reader(f)
        result_list = []

        for row in reader:

            result_list.append(row)

            multiple = 20

            if row[0] in multiple_dict.keys():
                multiple = multiple_dict[row[0]]

            for i in range(multiple):
                result_list.append([row[0].replace('.jpg', '_' + str(i) + '.jpg'), row[1]])

        with open('C:\\BoneAgeAssessment\\Route2\\dataset\\annos\\score1.csv', 'w', encoding='utf-8', newline='') as f:

            f.write(title)

            csv.writer(f).writerows(result_list)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

