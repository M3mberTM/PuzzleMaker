from descrambler.descrambler import Scramble


IMAGE = 'images/scrambledImages/black_white_test.png'
image_path = IMAGE
algorithm = Scramble(image_path, rows=9, cols=8)
