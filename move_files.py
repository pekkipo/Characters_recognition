import shutil
import os
import time

source = '../DR_data/capital_letters/'
dest = '../DR_data/characters_numbers/'


files = os.listdir(source)
start = time.time()
for f in files:
        shutil.copy(source+f, dest)

end = time.time()
print(end - start)