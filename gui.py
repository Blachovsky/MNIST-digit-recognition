from keras.models import load_model
import tkinter as tk
import win32gui
from PIL import ImageGrab, ImageOps
import numpy as np

model = load_model('mnist_model.h5')

def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    img = ImageOps.invert(img)
    img = img.convert('L')
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
        self.classify_pic_btn = tk.Button(self,text = "Recognize in picture", command=self.classify_digits_in_picture)
        self.classify_camera_btn = tk.Button(self,text = "Recognize with camera", command=self.classify_digits_with_camera) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky='w')
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.classify_btn.grid(row=1, column=1)
        self.classify_pic_btn.grid(row=2,column=1)
        self.classify_camera_btn.grid(row=3,column=1)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
    def classify_digits_in_picture(self):
        return
    def classify_digits_with_camera(self):
        return
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
    

app = App()

tk.mainloop()