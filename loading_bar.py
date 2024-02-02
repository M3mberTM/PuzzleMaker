import time
import sys

t="█"
a="."
for i in range(21): # way to make the progress bar work
    time.sleep(0.05)
    sys.stdout.write(f"\r[{t*i}{a*(100-i)}] | {i}/100")
    sys.stdout.flush()
sys.stdout.write(f"\r[{t*i}{a*(100-i)}] | {i}/100 Finished!")
sys.stdout.flush()

print()
print('bbbb')

