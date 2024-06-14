from os import listdir
from os.path import isfile, join
mypath = "images/testImages"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
    print(file)
