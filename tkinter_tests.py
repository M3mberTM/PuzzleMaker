import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join


class myGUI:
    images = []
    top_index = 0
    right_index = 0
    left_index = 0
    bottom_index = 0
    background_color = '#292f36'

    def __init__(self, images):
        self.images = images

        root = tk.Tk()
        placeholder_img = self.cv_to_pillow_img(self.images[0])

        root.geometry("800x800")  # dimensions of the window
        root.title = "Tests"  # title of the window
        root.configure(background=self.background_color)

        img_frame = tk.Frame(root, background=self.background_color)
        img_frame.columnconfigure(0, weight=2)
        img_frame.columnconfigure(1, weight=2)
        img_frame.columnconfigure(2, weight=3)
        img_frame.columnconfigure(3, weight=3)
        img_frame.columnconfigure(4, weight=3)
        img_frame.columnconfigure(5, weight=2)
        img_frame.columnconfigure(6, weight=2)

        btn_top_top = tk.Button(img_frame, text='^', bd=0, font=('Arial', 18),
                                command=lambda: self.change_btn_img(top_image_label, self.images[self.top_index], 'top',
                                                                    1))
        btn_top_top.grid(row=0, column=3, sticky='news')

        btn_top_bottom = tk.Button(img_frame, text='v', font=('Arial', 18), bd=0,
                                   command=lambda: self.change_btn_img(top_image_label, self.images[self.top_index],
                                                                       'top', -1))
        btn_top_bottom.grid(row=1, column=3, sticky='news')

        top_image_label = tk.Label(img_frame, image=placeholder_img, bd=0, bg=self.background_color)
        top_image_label.image = placeholder_img
        top_image_label.grid(row=2, column=3)

        btn_left_top = tk.Button(img_frame, text='<', font=('Arial', 18), bd=0,
                                 command=lambda: self.change_btn_img(left_image_label, self.images[self.left_index],
                                                                     'left', 1))
        btn_left_top.grid(row=3, column=0, sticky='news')

        btn_left_bottom = tk.Button(img_frame, text='>', font=('Arial', 18), bd=0,
                                    command=lambda: self.change_btn_img(left_image_label, self.images[self.left_index],
                                                                        'left', -1))
        btn_left_bottom.grid(row=3, column=1, sticky='news')

        left_image_label = tk.Label(img_frame, image=placeholder_img, bd=0)
        left_image_label.image = placeholder_img
        left_image_label.grid(row=3, column=2)

        center_image_label = tk.Label(img_frame, image=placeholder_img, bd=0)
        center_image_label.image = placeholder_img
        center_image_label.grid(row=3, column=3)

        right_image_label = tk.Label(img_frame, image=placeholder_img, bd=0)
        right_image_label.image = placeholder_img
        right_image_label.grid(row=3, column=4)

        btn_right_top = tk.Button(img_frame, text='<', font=('Arial', 18), bd=0,
                                  command=lambda: self.change_btn_img(right_image_label, self.images[self.right_index],
                                                                      'right', 1))
        btn_right_top.grid(row=3, column=5, sticky='news')

        btn_right_bottom = tk.Button(img_frame, text='>', font=('Arial', 18), bd=0,
                                     command=lambda: self.change_btn_img(right_image_label,
                                                                         self.images[self.right_index], 'right', -1))
        btn_right_bottom.grid(row=3, column=6, sticky='news')

        bottom_image_label = tk.Label(img_frame, image=placeholder_img, bd=0)
        bottom_image_label.image = placeholder_img
        bottom_image_label.grid(row=4, column=3)

        btn_bottom_top = tk.Button(img_frame, text='^', font=('Arial', 18), bd=0,
                                   command=lambda: self.change_btn_img(bottom_image_label,
                                                                       self.images[self.bottom_index], 'bottom', 1))
        btn_bottom_top.grid(row=5, column=3, sticky='news')

        btn_bottom_bottom = tk.Button(img_frame, text='v', font=('Arial', 18), bd=0,
                                      command=lambda: self.change_btn_img(bottom_image_label,
                                                                          self.images[self.bottom_index], 'bottom', -1))
        btn_bottom_bottom.grid(row=6, column=3, sticky='news')

        img_frame.pack()

        btn5 = tk.Button(root, text="Correct?", font=('Arial', 18), command=lambda c='Correct': print(c))
        btn5.pack(fill='x')

        root.mainloop()

    def cv_to_pillow_img(self, cv_image):
        pil_image = Image.fromarray(cv_image)
        imgTk = ImageTk.PhotoImage(pil_image)
        return imgTk

    def change_btn_img(self, parent, image, direction: str, increment_num: int):
        pil_img = self.cv_to_pillow_img(image)
        parent.configure(image=pil_img)
        parent.image = pil_img
        if direction == 'top':
            self.top_index = (self.top_index + increment_num) % len(self.images)
        elif direction == 'right':
            self.right_index = (self.right_index + increment_num) % len(self.images)
        elif direction == 'left':
            self.left_index = (self.left_index + increment_num) % len(self.images)
        elif direction == 'bottom':
            self.bottom_index = (self.bottom_index + increment_num) % len(self.images)


imagesPath = "pieces/"
onlyfiles = [f for f in listdir(imagesPath) if isfile(join(imagesPath, f))]

images = []
for path in onlyfiles:
    images.append(cv.imread(imagesPath + path, cv.IMREAD_COLOR))

gui = myGUI(images)
