import joblib
import pandas as pd
import numpy as np
from tkinter import *
import tkinter.ttk
from PIL import Image, ImageTk

myroot = Tk()

myroot.geometry('1100x900')
myroot.title("Used Car Price Prediction ")
myroot.resizable(False, False)
myimage = Image.open("images/img_1.png")
myimage_resize = myimage.resize((1100, 700), Image.Resampling.LANCZOS)
get_image = ImageTk.PhotoImage(myimage_resize)
my_root_label = Label(myroot, image=get_image, bd=0)
my_root_label.place(x=0, y=0)



#----------------Transmission label---------------
Transmissionl=Label(myroot,text='Transmission',fg='black',font='lucida 20 bold')
Transmissionl.place(x=10,y=80)
n = StringVar()
Transmission_choose = tkinter.ttk.Combobox(myroot, width=20, height=200, textvariable=n)
Transmission_choose['value'] = ('Select Transmission', 'manual', 'automatic')
Transmission_choose.current(0)
Transmission_choose.place(x=200, y=80)

#----------------Fuel label---------------
Fuell=Label(myroot,text='Fuel',fg='black',font='lucida 20 bold')
Fuell.place(x=10,y=120)
m = StringVar()
Fuel_choose = tkinter.ttk.Combobox(myroot, width=20, height=400, textvariable=m)
Fuel_choose['value'] = ('Fuel Type', 'diesel', 'petrol','lpg','cng')
Fuel_choose.current(0)
Fuel_choose.place(x=200, y=120)

#----------------Owner label---------------
Ownerl=Label(myroot,text='owner',fg='black',font='lucida 20 bold')
Ownerl.place(x=10,y=160)
o = StringVar()
Owner_choose = tkinter.ttk.Combobox(myroot, width=20, height=400, textvariable=o)
Owner_choose['value'] = ('Owner', 'first owner', 'second owner','third owner')
Owner_choose.current(0)
Owner_choose.place(x=200, y=160)

#----------------Seller label---------------
Seller_typel=Label(myroot,text='Seller type',fg='black',font='lucida 20 bold')
Seller_typel.place(x=10,y=200)
p = StringVar()
Seller_choose = tkinter.ttk.Combobox(myroot, width=20, height=400, textvariable=p)
Seller_choose['value'] = ('Seller Type', 'individual', 'dealer','trustmark dealer')
Seller_choose.current(0)
Seller_choose.place(x=200, y=200)

#----------------Seats label---------------
Seatsl=Label(myroot,text='Seats',fg='black',font='lucida 20 bold')
Seatsl.place(x=10,y=240)
q = StringVar()
Seats_choose = tkinter.ttk.Combobox(myroot, width=20, height=400, textvariable=q)
Seats_choose['value'] = ('Seats', '2', '4','5','7','8')
Seats_choose.current(0)
Seats_choose.place(x=200, y=240)

#----------------Engine label-----------------
Enginel=Label(myroot,text='Engine',fg='black',font='lucida 18 bold')
Enginel.place(x=10,y=280)
Engine_entry = Entry(myroot, fg="blue", font="rockwell 18 bold")
Engine_entry.place(x=250, y=280, width=200, height=35)

#----------------Power label-----------------
Powerl=Label(myroot,text='Power',fg='black',font='lucida 18 bold')
Powerl.place(x=10,y=320)
Power_entry = Entry(myroot, fg="black", font="rockwell 18 bold")
Power_entry.place(x=250, y=320, width=200, height=35)

#----------------Year label-----------------
Yearl=Label(myroot,text='Year',fg='black',font='lucida 18 bold')
Yearl.place(x=10,y=360)
Year_entry = Entry(myroot, fg="red", font="rockwell 18 bold")
Year_entry.place(x=250, y=360, width=200, height=35)

#----------------mileage label-----------------
Mileagel=Label(myroot,text='Mileage',fg='black',font='lucida 18 bold')
Mileagel.place(x=10,y=400)
Mileage_entry = Entry(myroot, fg="green", font="rockwell 18 bold")
Mileage_entry.place(x=250, y=400, width=200, height=35)

#----------------Km label-----------------
Km_drivenl=Label(myroot,text='Kilometers driven',fg='black',font='lucida 18 bold')
Km_drivenl.place(x=10,y=440)
Km_driven_entry = Entry(myroot, fg="yellow", font="rockwell 18 bold")
Km_driven_entry.place(x=250, y=440, width=200, height=35)


#------------load joblib files ---------
la=joblib.load('transmission.joblib')
li=joblib.load('fuel.joblib')
le=joblib.load('seller_type.joblib')
ct=joblib.load('onehot.joblib')
sc=joblib.load('scaler.joblib')
regressor=joblib.load('regressor.joblib')

def value():
 transmission = Transmission_choose.get()
 fuel = Fuel_choose.get()
 owner = Owner_choose.get()
 seller_type = Seller_choose.get()
 seats = Seats_choose.get()
 engine = Engine_entry.get()
 max_power = Power_entry.get()
 year = Year_entry.get()
 mileage=Mileage_entry.get()
 km_driven=Km_driven_entry.get()

 data = pd.DataFrame({'transmission': [transmission], 'fuel': [fuel], 'owner': [owner], 'seller_type': [seller_type],
                      'seats': [seats], 'year': [year], 'mileage': [mileage], 'km_driven': [km_driven],
                      'max_power': [max_power], 'engine': [engine]})
 data['transmission'] = la.transform(data['transmission'])
 data['fuel'] = li.transform(data['fuel'])
 data['seller_type'] = le.transform(data['seller_type'])
 data = ct.transform(data)
 data = sc.transform(data)
 label = Label(myroot, text=regressor.predict(data),bg="silver",font="lucida 20 bold")
 label.place(x=430,y=450)
 my_value = Label(myroot, text="Your Car is Worth  ", fg="black", bg="silver", font="lucida 20 bold ")
 my_value.place(x=420, y=500)

#---------Predict Image-------------
myimage3 = Image.open("images/img_2.png")
myimage_resize_newuser2 = myimage3.resize((160, 70), Image.Resampling.LANCZOS)
get_image_newuser2 = ImageTk.PhotoImage(myimage_resize_newuser2)
button_newuser2 = Button(myroot, image=get_image_newuser2, cursor="hand2", command=value, borderwidth=0)
button_newuser2.place(x=150, y=500)

#---------------Title label-----------
my_title = Label(myroot, text="Used Car Price prediction ", fg="black", bg="silver", font="lucida 20 bold ")
my_title.place(x=380, y=50)
myroot.mainloop()
