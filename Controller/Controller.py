import gi
from typing import Any


import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf,GLib
import cv2
from Model import Mixed, ORB, AKAZE, BRISK
import urllib, json
import requests

global builderOne
global builderTwo
global onClick
global SampeInput





onClick = False
cap = cv2.VideoCapture(0)

builderOne = Gtk.Builder()
builderTwo = Gtk.Builder()
builderOne.add_from_file("/home/nick/Downloads/BackStoryMVC/View/ViewXML.glade")
builderTwo.add_from_file("/home/nick/Downloads/BackStoryMVC/View/glade.glade")
image = builderTwo.get_object("image")



def getFeatureDetector(frame, featureDetector):


    if featureDetector is "BRISK":
        brisk = BRISK.BRISK(frame, "/home/nick/Downloads/BackStoryMVC/Model/Database")
        return brisk
    if featureDetector is "ORB":
        orb = ORB.ORB(frame, "/home/nick/Downloads/BackStoryMVC/Model/testSamples")
        return orb
    if featureDetector is "AKAZE":
        akaze = AKAZE.AKAZE(frame, "/home/nick/Downloads/BackStoryMVC/Model/testSamples")
        return akaze
    else:
        mixed = Mixed.MIXED(frame, "/home/nick/Downloads/BackStoryMVC/Model/testSamples")
        return mixed


def getWindows(windowName):
    if windowName is "mainWindow":
        mainWindow = builderOne.get_object("MainWindow")
        return mainWindow
    if windowName is "actionWindow":
        actionWindow = builderOne.get_object("ActionWindow")
        return actionWindow
    if windowName is "snapShotWindow":

        snapShotWindow = builderTwo.get_object("SnapShotWindow")
        return snapShotWindow
    else:
        print("invalid name")





def getSampleImage(frame):
    frame = cv2.resize(frame,None,fx=0.5,fy=0.5)
    background = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pb = GdkPixbuf.Pixbuf.new_from_data(background.tostring(),
                                        GdkPixbuf.Colorspace.RGB,
                                        False,
                                        8,
                                        background.shape[1],
                                        background.shape[0],
                                        background.shape[2] * background.shape[1])
    return pb.copy()

def getDetectedImage(frame):

    detector = getFeatureDetector(frame, "BRISK")
    potentialImage = detector.finalImage
    background = potentialImage
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    pb = GdkPixbuf.Pixbuf.new_from_data(background.tostring(),
                                           GdkPixbuf.Colorspace.RGB,
                                           False,
                                           8,
                                           background.shape[1],
                                           background.shape[0],
                                           background.shape[2] * background.shape[1])
    return pb.copy()
def fetchData(objectId):
    response = requests.get('http://localhost:8080/v1/object/'+objectId)
    return response.json()

def updateLabels():
    data = fetchData("437742")
    title = builderOne.get_object("titleResponse")
    title.set_text(data["title"])
    title = builderOne.get_object("artistResponse")
    title.set_text(data["name"])
    title = builderOne.get_object("periodResponse")
    title.set_text(data["period"])
    title = builderOne.get_object("objectDateResponse")
    title.set_text(data["objectDate"])

def show_frame(*args):

    ret, frame = cap.read()


    if onClick:

        sampleImage = builderOne.get_object("SampleImage")
        detectedImage = builderOne.get_object("DetectedResult")

        sampleImage.set_from_pixbuf(getSampleImage(frame))
        detectedImage.set_from_pixbuf(getDetectedImage(frame))
        updateLabels()
        cap.release()

        cv2.destroyAllWindows()


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pb = GdkPixbuf.Pixbuf.new_from_data(frame.tostring(),
                                        GdkPixbuf.Colorspace.RGB,
                                        False,
                                        8,
                                        frame.shape[1],
                                        frame.shape[0],
                                        frame.shape[2] * frame.shape[1])

    image.set_from_pixbuf(pb.copy())

    return True



class Handler:
    global mainWindow
    global snapShotWindow
    global actionWindow


    snapShotWindow = getWindows("snapShotWindow")
    actionWindow = getWindows("actionWindow")


    def continueClicked(self, *args):

        GLib.idle_add(show_frame)
        mainWindow.destroy()


        snapShotWindow.show()



    def onDeleteWindow(self, *args):
        Gtk.main_quit(*args)


    def onDeleteWindowSnapShot(self, *args):
        Gtk.main_quit(*args)



    def toggleSnapShot(self, *args):
        global onClick
        onClick = ~ onClick
        snapShotWindow.destroy()
        actionWindow.show()



mainWindow = getWindows("mainWindow")


mainWindow.show()
builderTwo.connect_signals(Handler())
builderOne.connect_signals(Handler())
Gtk.main()