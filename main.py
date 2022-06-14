from flask import Flask, render_template, request, flash, Markup
import tkinter as tk
import prediction


#Creating tkinter GUI
import tkinter
from tkinter import *
p=prediction.predict()

#root window
root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
    
        ints = p.predict_class(msg)
        res = p.response(ints)
        
        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
 



#Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatBox.config(state=DISABLED)

#Create Button to send message
SendButton = Button(root, font=("Arial",13,'bold'), text="Send", width="10", height=3,
                    bd=0, bg="lightskyblue", activebackground="steelblue",command= send )

#Create the box to enter message
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")


#Place all components on the screen
ChatBox.place(x=6,y=6, height=386, width=385)
EntryBox.place(x=6, y=401, height=90, width=285)
SendButton.place(x=286, y=401,height=90)

root.mainloop()