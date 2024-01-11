from tkinter import Tk # from tkinter import Tk for Python 3.x
from os import listdir
from sys import argv
from tkinter.filedialog import askdirectory


print("APP START")
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
input_dir = askdirectory() # show an "Open" dialog box and return the path to the selected file
output_dir = askdirectory()
for file in listdir(input_dir):
    print(file)
print(input_dir)
print(output_dir)
print(str(argv))
