from descrambler.scramble import Scramble


IMAGE = 'images/testImages/08_851131_851131_1_006_001.jpg'
image_path = IMAGE
algorithm = Scramble(image_path, rows=5, cols=8)
algorithm.solve()