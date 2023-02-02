from keras.models import load_model
import tkinter as tk
import win32gui
from PIL import ImageGrab, ImageOps, ImageEnhance
import numpy as np


model = load_model('models/my_data_model.h5')

def resize_and_center(sample, new_size=28):
    inv_sample = ImageOps.invert(sample)
    bbox = inv_sample.getbbox()
    crop = inv_sample.crop(bbox)
    delta_w = new_size - crop.size[0]
    delta_h = new_size - crop.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(crop, padding)

def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.convert(mode='L')
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.resize((28,28))
    img = resize_and_center(img)
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img.astype('float32')
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise in canvas", command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky='w')
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.classify_btn.grid(row=1, column=1)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        rect = list(rect)
        rect[0] = rect[0] + 4
        rect[1] = rect[1] + 4
        rect[2] = rect[2] - 4
        rect[3] = rect[3] - 4
        rect = tuple(rect)
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
    

app = App()

tk.mainloop()