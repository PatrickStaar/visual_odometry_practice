from tkinter import *
import tkinter.filedialog as fdg
from PIL import Image, ImageTk
from eval import Eval
import cv2
import numpy as np

color = '#4A708B'


class App():

    def __init__(self, master=None):
        self.img0 = None
        self.img1 = None
        self.img_mono = None
        self.video = None

        self.tkimg0 = None
        self.tkimg1 = None
        self.tkimg2 = None
        self.tkimg3 = None

        self.eval_setup()

        self.window = Tk()
        self.window_init()
        self.size = self.window.size()
        self.panel = Frame(master=self.window, bg=color)
        self.viewer = Frame(master=self.window)

        self.une_input = Frame(master=self.viewer)
        self.deux_inputs = Frame(master=self.viewer)
        self.video_input = Frame(master=self.viewer)

        # Panel
        self.logo = Label(master=self.panel, bg=color, text='Visual Odometry', relief='flat', font='Arial', fg='white')
        self.b_vo = Button(master=self.panel, bg=color, text='里程计', relief='flat', activebackground='white', fg='white')
        self.b_dep = Button(master=self.panel, bg=color, text='单目深度预测', relief='flat', activebackground='white',
                            fg='white')
        self.b_vid = Button(master=self.panel, bg=color, text='视频分析', relief='flat', activebackground='white',
                            fg='white')

        # Viewer
        ## Buttons
        self.b_select = Button(master=self.une_input, text='选择一张图片', relief='flat', activebackground=color)
        self.b_select_1 = Button(master=self.deux_inputs, text='选择Frame0', relief='flat', activebackground=color)
        self.b_select_2 = Button(master=self.deux_inputs, text='选择Frame1', relief='flat', activebackground=color)
        self.b_select_3 = Button(master=self.video_input, text='选择一个视频', relief='flat', activebackground=color)

        self.b_run_0 = Button(master=self.une_input, text='启动单目深度预测', relief='flat', activebackground=color)
        self.b_run_1 = Button(master=self.deux_inputs, text='启动里程计', relief='flat', activebackground=color)
        self.b_run_2 = Button(master=self.video_input, text='启动视频分析', relief='flat', activebackground=color)

        # Canvases
        self.c_mono = Canvas(master=self.une_input, )
        self.c_deux_1 = Canvas(master=self.deux_inputs, )
        self.c_deux_2 = Canvas(master=self.deux_inputs, )
        self.c_video_3 = Canvas(master=self.video_input, )

        self.layout()
        self.events()

    def imopen(self, event):

        target = event.widget['text']
        img = None
        filename = fdg.askopenfilename()
        if 'mp4' in filename:
            cap = cv2.VideoCapture(filename)
            ok, frame = cap.read()
            cap.release()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(np.uint8(frame))
        else:
            img = Image.open(filename)

        img = img.resize((640, 480))
        tkimg = ImageTk.PhotoImage(image=img.resize((320, 240)))

        if target == '选择一张图片':
            self.tkimg0 = tkimg
            self.c_mono.create_image(30, 10, anchor='nw', image=self.tkimg0)
            self.img_mono = img
        elif target == '选择Frame0':
            self.tkimg1 = tkimg
            self.c_deux_1.create_image(30, 10, anchor='nw', image=self.tkimg1)
            self.img0 = img
        elif target == '选择Frame1':
            self.tkimg2 = tkimg
            self.c_deux_2.create_image(30, 10, anchor='nw', image=self.tkimg2)
            self.img1 = img
        else:
            self.tkimg3 = tkimg
            self.video = filename
            self.c_video_3.create_image(30, 10, anchor='nw', image=self.tkimg3)

    def layout(self):

        # Panel
        self.panel.place(relx=0, rely=0, relwidth=0.25, relheight=1)

        self.viewer.place(relx=0.25, y=0, relwidth=0.75, relheight=1)

        self.une_input.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.deux_inputs.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.video_input.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.logo.place(x=0, y=0, relwidth=1, height=50)
        self.b_vo.place(x=0, y=50, relwidth=1, height=50)
        self.b_dep.place(x=0, y=100, relwidth=1, height=50)
        self.b_vid.place(x=0, y=150, relwidth=1, height=50)

        self.b_select.place(relx=0, y=0, relwidth=0.3, height=50)
        self.b_select_1.place(relx=0, y=0, relwidth=0.3, height=50)
        self.b_select_2.place(relx=0, y=50, relwidth=0.3, height=50)
        self.b_select_3.place(relx=0, y=0, relwidth=0.3, height=50)

        self.b_run_0.place(x=0, y=50, relwidth=0.3, height=50)
        self.b_run_1.place(x=0, y=100, relwidth=0.3, height=50)
        self.b_run_2.place(x=0, y=50, relwidth=0.3, height=50)

        # Canvases
        self.c_mono.place(relx=0.3, y=20, relwidth=0.7, height=250)
        self.c_deux_1.place(relx=0.3, y=20, relwidth=0.7, height=250)
        self.c_deux_2.place(relx=0.3, y=300, relwidth=0.7, height=250)
        self.c_video_3.place(relx=0.3, y=20, relwidth=0.7, height=250)

    def events(self):
        self.b_vo.bind('<Button-1>', self.show_frame)
        self.b_dep.bind('<Button-1>', self.show_frame)
        self.b_vid.bind('<Button-1>', self.show_frame)

        self.b_select.bind('<Button-1>', self.imopen)
        self.b_select_1.bind('<Button-1>', self.imopen)
        self.b_select_2.bind('<Button-1>', self.imopen)
        self.b_select_3.bind('<Button-1>', self.imopen)

        self.b_run_0.bind('<Button-1>', self.process)
        self.b_run_1.bind('<Button-1>', self.process)
        self.b_run_2.bind('<Button-1>', self.process)

    def window_init(self):
        self.window.title('Visual Odometry End-to-End')
        width, height = self.window.maxsize()
        self.window.geometry("{}x{}".format(width // 2, height // 2))
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)

    def show_frame(self, event):
        target = event.widget['text']
        if target == '里程计':
            self.deux_inputs.tkraise()
        elif target == '单目深度预测':
            self.une_input.tkraise()
        else:
            self.video_input.tkraise()

    def process(self, event):
        self.output = {}
        target = event.widget['text']
        if '深度' in target:
            self.output = self.eval.depth_estimate(self.img_mono)

        elif '里程计' in target:
            self.output = self.eval.pred(self.img0, self.img1)
        else:
            self.output = self.eval.pred_video(self.video)

        self.eval.visual(self.output)

    def eval_setup(self):
        x = Eval()
        x.load_model()
        x.load_depth_model()
        self.eval = x


if __name__ == '__main__':
    app = App()
    # to do
    app.window.mainloop()
