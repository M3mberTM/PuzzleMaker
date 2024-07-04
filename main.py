from descrambler.descrambler import Scramble


IMAGE = 'images/testImages/08_851131_851131_1_006_001.jpg'
image_path = IMAGE
algorithm = Scramble(image_path, rows=10, cols=10)
algorithm.solve()