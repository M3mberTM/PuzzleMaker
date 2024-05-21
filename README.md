# Descrambler and scrambler (Puzzle maker)

Written in Python 3.9. App for descrambling of Scanlation images, although it can also be used for simple puzzles.

## Scrambler

Takes an image and divided it into x rows and y columns based on the arguments given.

## Descrambler

Tries to fix the scrambled images back into original. For now, it solves it by comparing the corner values of each
piece.

How does it work?

- [ ] Tkinter interface where you can select the starting files and ending directory
- [ ] Confirmation button for the directories
- [ ] Pop up for information:
    - [ ] number of rows
    - [ ] number of columns
    - [ ] checkbox (is it the same for all the files?)
- [ ] Get all the pieces from the images
- [ ] Turn them into quadrants
- [ ] Tkinter interface for adjusting the quadrants of each image
- [ ] Final quadrants turn into final image

### Selecting files and ending directory

Simple interface with button for selecting the files and an ending directory. There should be a text field that displays
the selected directories.

Show text inputs for numbers that give number of rows and columns. Also include a checkbox that asks if the number of
columns and rows is the same for all the pictures. If yes, ignore. If no, make some sort of menu for specifying the
number of rows and columns for each image.

After selecting the directories, there should be a confirmation button for the given directories!

### Getting the pieces

Get all the pieces from the image and put them into quadrants. We can assume that these quadrants will not be completely
accurate and therefore will need manual adjustments.

### Manual adjustment of the quadrants

Tkinter interface that allows for switching pieces in the quadrants and easy editing. Should contain indicators of which
piece is which, which is the current quadrant and buttons for easy switching between piece on each side.

### Final image

We can assume that if all the quadrants are correct, the final image is easy to make. Select random quadrant as the
center of the final image. From this center quadrant, on the x axis from each side x pieces will be added from other
quadrants, where x is number of columns - 1. At the same time, on the y axis from each side, y pieces will be added,
where y is number of rows -1. The final image will be twice the size of the wanted image. The actual image will have to
be cropped out of the final image.