# Utility script to visualize a sequence of image files:
# python Movie.py name*.*
#
# For example: python Movie.py stage*.ppm
# 

import tkinter
import time

animation_window_width = 800
animation_window_height = 600
animation_refresh_seconds = 0.1
num_image = 0
image_width = 0
image_height = 0

import sys

if len(sys.argv) < 2:
   print("To execute:\n python Movie.py name*.*\n")
   sys.exit()
else:
   nimage0 = sys.argv[1]
   
import os
nimage = os.path.splitext(nimage0)[0].strip("0123456789")
fmt = os.path.splitext(nimage0)[1]
import glob
nimage_l = glob.glob(nimage+"*"+fmt)
mximage = len(nimage_l)
if mximage <= 0:
   print("There are no files with name %s*.*\n"%nimage)
   sys.exit()
nimage_l.sort()

def create_animation_window():
   window = tkinter.Tk()
   window.title("Visualizing")
   window.geometry(f'{animation_window_width}x{animation_window_height}')
   return window
   
def create_animation_canvas(window):
   wrap = tkinter.Canvas(window)
   wrap.configure(bg="blue")
   wrap.pack(fill="both",expand=True)
   return wrap

import PIL as P
import PIL.Image as Image
import PIL.ImageTk as ImageTk

def TKimage(fn):
   global animation_window_width, animation_window_height, image_width, image_height

   try:
      PILimg=P.Image.open(fn)
   except:
      print("In TKimage error reading image %s."%fn)
   image_width = PILimg.size[0]
   image_height = PILimg.size[1]
   try:
      PHOimg=ImageTk.PhotoImage(PILimg)  # convert Image object into PhotoImage object
   except:
      print("In TKimage error ImageTk.PhotoImage(%s)."%fn)
   return PHOimg
#end TKimage(fn)
   

def normalize(fn,nfn):
   global animation_window_width, animation_window_height, image_width, image_height
   try:
      PILimg=P.Image.open(fn)
   except:
      print("In normalize() error reading image %s."%fn)

   Larg = animation_window_width
   Alt = animation_window_height
   image_width = PILimg.size[0]
   image_height = PILimg.size[1]
   if PILimg.size[0] > PILimg.size[1]:
      a=int(Larg*PILimg.size[1]/PILimg.size[0])
      ns=(Larg,a)
   else:
      l=int(Alt*PILimg.size[0]/PILimg.size[1])
      ns=(l,Alt)
   try:
      nimg=PILimg.resize(ns)
   except:
      print(f"In normalize() error PILimg.resize({ns})")
      return
   nimg.save(nfn)
#end normalize(fn,nfn)
   
def visualize(window, canvas):
   global nimage_l, num_image, Previous

   while True:
      time.sleep(animation_refresh_seconds)
      canvas.delete('currimg')
      img = TKimage(normimage_l[num_image])
      num_image = (num_image+1)%mximage
      x0 = int(( animation_window_width - image_width )/2.0)
      y0 = int(( animation_window_height - image_height )/2.0)
      canvas.create_image(x0,y0,anchor=tkinter.NW,image=img,tag='currimg')
      canvas.img=img
      canvas.update_idletasks()
      window.update()
   
normimage_l = list()

for nimage in nimage_l:
   nome,ext = os.path.splitext(nimage)
   normimage = f"norm{nome}.png"
   normimage_l.append(normimage)
   if not os.path.isfile(normimage): # non ancora elaborato
      normalize(nimage,normimage)
  
animation_window = create_animation_window()
animation_canvas = create_animation_canvas(animation_window)

visualize(animation_window,animation_canvas)

