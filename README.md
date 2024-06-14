# Descrambler and scrambler (Puzzle maker)

Written in Python 3.9. App for descrambling of Scanlation images, although it can also be used for simple puzzles.

## Scrambler

Takes an image and divided it into x rows and y columns based on the arguments given.

## Descrambler

Takes a scrambled image and unscrambles it.


### Problems

#### Image cutting

Problem:
>In the beginning, the image is divided into x rows and y columns. These variables are given by the user in the
beginning. After multiple problems with the algorithm, several pieces seem to have been cut wrong. At least, that is the
way they seemed.

The debugging consisted of taking the pieces the moment they are cut and putting them back into the original order of
the image to see if there are some irregularities. ***There didn't seem to be any though so the issue lies elsewhere!***

#### False positives

In the initial part of putting pieces next to each other, the original algorithm seems to run into few false positives.
I am not quite sure why some of them are the best edges instead of the real one but, the way to circumvent this would be
to also compare the edges to more than one piece to get more of an general solution.

### Algorithm

#### Dividing of pieces

Given a user input, the image is split into x rows and y columns, where x and y are given by user as user input. These
are translated into into a Piece object which contains the original piece as well as a grayscale version for easier work
in the algorithm

#### Getting main edges

The first function gets the most fitting edges for all the pieces and gives them a fitness value. These are then sorted
so the lowest fitness value is the best piece

