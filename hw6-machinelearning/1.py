#TODO
#bug: start polygon drawing, dont right click the last line, it still shows, but isnt linked to the whole polygon

#IMPORTED LIBRARIES
###################

from tkinter import *
from tkinter import ttk
import copy
# import dill
import pickle
import os

#DATA INITIALIZATION
####################

lastx, lasty = 0, 0			#to keep some initial point coordinates
previous_object=-1			#the last object operated on
previous_polygon=-1			#the last polygon operated on
color = "black"				#color currently selected
figure = "freehand"			#draw figure mode currently selected
polygon = []				#coordinates for current polygon being drawn
mode = "draw"				#main mode of the program ("draw","select")
select_mode = "move"		#select specific mode of the program("move", "cut", "copy", "paste")
selection_id = -1			#id of selected object
clipboard_info=[]			#to store info about copy/cut item
advanced_mode="group"		#set advanced mode("group", "ungroup", "undo", "redo")
group_elements=[]			#list of id of elements to group
counter_obj={"group":0,"freehand":0,"polygon":0}
debugOutput=False			#set debug level

#CUSTOM FUNCTIONS
#################

def sign(number):
	if number>0:
		return 1;
	elif number<0:
		return -1;
	else:
		return 0;

def delObject(toDelete):										#delete object if not UI
	if toDelete != -1:
		canvas.delete(toDelete)

def findNearest(event):
	default_id = canvas.find_closest(event.x, event.y)[0]
	selection_tags = canvas.gettags(default_id)
	for tag in selection_tags:									#iterate over all tags
		if tag.find("button") != -1:							#if button tag
			return -1											#return invalid
	for tag in selection_tags:									#iterate over all tags
		if tag.find("group") != -1:								#if group tag
			return tag											#return group tag
	for tag in selection_tags:									#iterate over all tags
		if (tag.find("free")!=-1) or (tag.find("poly")!=-1):	#if custom tag
			return tag											#return custom tag
	return default_id											#else return the default returned id	

def tagsBeforeGroup(identifier):
	allTags = canvas.gettags(identifier)
	returnTags=[]
	for j in allTags:
		if j.find("group")==-1:									#if not group tag in all tags
			returnTags.append(j)								#return tag
	return returnTags											#else return empty array

def getNewTag(tag):
	global counter_obj
	tagType,suffix=tag.split('_')
	counter_obj[tagType]=counter_obj[tagType]+1
	newTag=tagType+"_"+str(counter_obj[tagType])
	return newTag

#DATA SETTING FUNCTIONS
#######################

def setColor(newcolor):
    global color
    color = newcolor
    print("Color: "+color)
    
def setFigure(newfigure):
	global figure, mode
	figure=newfigure
	mode="draw"
	print("Mode: Draw ; Figure: "+figure)

def setSelection(newMode):
	global figure, mode, select_mode, selection_id
	mode="select"
	select_mode = newMode
	selection_id = -1
	print("Mode: Select ; Option: "+select_mode)

def setAdvanced(newMode):
	global figure, mode, advanced_mode, selection_id, group_elements
	mode="advanced"
	group_elements = []
	advanced_mode = newMode
	selection_id = -1
	print("Mode: Advanced ; Option: "+advanced_mode)

def setSaved(option):
	if option == "save":
		saveToDisk()
		print("Saving to disk")
	elif option == "load":
		loadFromDisk()
		print("Loading from disk")
	else:
		print("ERROR: INVALID ELSE 6")

#DRAWING RELATED FUNCTIONS
##########################

def addFree(event):													#create a new line, from last point to current, and tag it
    global lastx, lasty, counter_obj
    prefix='freehand_'
    canvas.create_line((lastx, lasty, event.x, event.y), fill=color, tags=(prefix+str(counter_obj["freehand"])))
    lastx, lasty = event.x, event.y
    return prefix+str(counter_obj["freehand"])

def addStraight(event):
    global lastx, lasty, previous_object
    delObject(previous_object)
    previous_object = canvas.create_line((lastx, lasty, event.x, event.y), fill=color)
    
def addEllipse(event):
    global lastx, lasty, previous_object
    delObject(previous_object)
    previous_object = canvas.create_oval((lastx, lasty, event.x, event.y), fill=color)

def addRectangle(event):
    global lastx, lasty, previous_object
    delObject(previous_object)
    previous_object = canvas.create_rectangle((lastx, lasty, event.x, event.y), fill=color)
    
def addSquare(event):
    global lastx, lasty, previous_object
    delObject(previous_object)
    square_dim = min(abs(event.x-lastx), abs(event.y-lasty))	#find max size square in current selection
    newx = lastx + (square_dim*sign(event.x-lastx))
    newy = lasty + (square_dim*sign(event.y-lasty))
    previous_object = canvas.create_rectangle((lastx, lasty, newx, newy), fill=color)

def addCircle(event):
    global lastx, lasty, previous_object
    delObject(previous_object)
    circle_dim = min(abs(event.x-lastx), abs(event.y-lasty))	#find max size square in current selection
    newx = lastx + (circle_dim*sign(event.x-lastx))
    newy = lasty + (circle_dim*sign(event.y-lasty))
    previous_object = canvas.create_oval((lastx, lasty, newx, newy), fill=color)

#POLYGON SPECIFIC DRAWING FUNCTIONS

def addPolygon(event):									#rubberbanding polygon line, on mouse move
    global lastx, lasty, previous_object
    delObject(previous_object)
    previous_object = canvas.create_line((lastx, lasty, event.x, event.y), fill=color, width=2)

def drawPolygon(polygondata, outline, width):			#create a polygon with custom tag, and multiple lines, on right click
	global counter_obj
	canvas.delete(previous_object)
	prefix='polygon_'
	counter_obj["polygon"] = counter_obj["polygon"]+1
	for i in range(len(polygondata)-1):
		id = canvas.create_line(polygondata[i],polygondata[i+1], width=width, fill=outline, tags=(prefix+str(counter_obj["polygon"])))
	return prefix+str(counter_obj["polygon"])

def cleanPolygon(polygondata, options):				#doesnt delete anything, creates a new polygon
	global counter_obj
	prefix='polygon_'
	counter_obj["polygon"] = counter_obj["polygon"]+1
	for i in polygondata:
		id = canvas.create_line(i)
		canvas.itemconfigure(id, options)
		canvas.dtag(id)
		canvas.itemconfig(id, tags=(prefix+str(counter_obj["polygon"])))
	return prefix+str(counter_obj["polygon"])

#MOVEMENT FUNCTIONS
###################

def selectOptions(event):
	global selection_id, lastx, lasty
	if select_mode == "cut":
		retainObjectInfo(event)
		canvas.delete(selection_id)
	elif select_mode == "copy":
		retainObjectInfo(event)
	elif select_mode == "move":
		lastx, lasty = event.x, event.y
	elif select_mode == "paste":
		pass
	else:
		print("ERROR:WRONG MODE DETECTED 5")

def moveStuff(event):
	global lastx, lasty
	if selection_id != -1:									#dont move if invalid selection_id(buttons)
		xAmount = event.x-lastx
		yAmount = event.y-lasty
		canvas.move(selection_id, xAmount, yAmount)
		lastx, lasty = event.x, event.y

def retainObjectInfo(event):
	global selection_id, clipboard_info
	clipboard_info=[]										#array to store current clipboard drawings
	for drawing_id in canvas.find_withtag(selection_id):	#iterate over drawings with selection_id
		drawing_info={}										#object to store each drawing info
		drawing_info["options"]={}
		drawing_info["type"]=canvas.type(drawing_id)
		drawing_info["options"]["width"]=canvas.itemcget(drawing_id, "width")
		drawing_info["options"]["fill"]=canvas.itemcget(drawing_id, "fill")
		drawing_info["options"]["tags"]=canvas.itemcget(drawing_id, "tags")
		drawing_info["coords"]=canvas.coords(drawing_id)
		clipboard_info.append(drawing_info)					#append drawing to clipboard object
	if debugOutput:
		print(clipboard_info)
		
def renderClipboard(event):
	global clipboard_info
	function_list={
		"freehand":cleanPolygon, 
		"line":getattr(canvas,"create_line"),
		"rectangle":getattr(canvas,"create_rectangle"),
		"oval":getattr(canvas,"create_oval"),
		"polygon":cleanPolygon
	}
	tags_seen={}
	clip=copy.deepcopy(clipboard_info)
	if not clip:													#if no copied data
		return														#return
	xDiff=event.x-clip[0]["coords"][0]
	yDiff=event.y-clip[0]["coords"][1]
	for drawing in clip:
		drawing["coords"][0]=drawing["coords"][0]+xDiff
		drawing["coords"][2]=drawing["coords"][2]+xDiff
		drawing["coords"][1]=drawing["coords"][1]+yDiff
		drawing["coords"][3]=drawing["coords"][3]+yDiff
		if drawing["options"]["tags"]!='':									#check if any tags exist
			old_tags = drawing["options"]["tags"].split(" ")				#get tags and split them by space
			new_tags=[]
			for curr_tag in old_tags:
				if curr_tag not in ("{}","current"):						#avoid corner cases
					if curr_tag not in tags_seen:							#if tag isnt seen before
						tags_seen[curr_tag]=getNewTag(curr_tag)				#get new tag
					new_tags.append(tags_seen[curr_tag])					#append new/generated tag from object
			drawing["options"]["tags"] = new_tags
		function_list[drawing["type"]](drawing["coords"], drawing["options"])

#ADVANCED FEATURE RELATED FUNCTIONS
###################################

def advancedOptions(event):
	global advanced_mode, lastx, lasty, selection_id, group_elements
	if advanced_mode == "group":
		selection_id = findNearest(event)
		if selection_id!=-1:
			group_elements.append(selection_id)
	elif advanced_mode == "ungroup":
		selection_id = findNearest(event)
		ungroupElements(event)
	else:
		print("ERROR:WRONG MODE DETECTED 1")
	
def groupElements(event):
	global counter_obj
	prefix="group_"
	counter_obj["group"] = counter_obj["group"] + 1
	for i in group_elements:
		canvas.itemconfig(i, tags=(prefix+str(counter_obj["group"]),canvas.gettags(i)))
	print("Group created: "+prefix+str(counter_obj["group"]))

def ungroupElements(event):
	global selection_id
	if type(selection_id)==int:
		return
	if selection_id.find("group")!=-1:						#if the id of the selected object is of group type
		for i in canvas.find_withtag(selection_id):			#iterate over all objects with group tag
			canvas.itemconfig(i, tags=tagsBeforeGroup(i))	#only store the second tag of the element
	print("Group ungrouped: "+selection_id)

#VERY ADVANCED(SAVE AND LOAD)
#############################

def saveToDisk(): 											#get info of all objects, and of the counter object, remove buttons, save the object to disk in "datafile"
	save_object={}
	save_object["counter_obj"]=counter_obj
	save_object["objects"]=[]
	
	items = canvas.find_all()
	for curr_item in items:
		tags_list=canvas.itemcget(curr_item, "tags").split(" ")
		flag=False
		for curr_tag in tags_list:						#check all tags for button object
			if curr_tag.find("buttons")!=-1:
				flag=True
		if not flag:									#if its not a buttons object
			item_info={}								#object to store each item info
			item_info["options"]={}
			item_info["type"]=canvas.type(curr_item)
			item_info["options"]["width"]=canvas.itemcget(curr_item, "width")
			item_info["options"]["fill"]=canvas.itemcget(curr_item, "fill")
			item_info["options"]["tags"]=tags_list
			item_info["coords"]=canvas.coords(curr_item)
			save_object["objects"].append(item_info)	#append item data to saving object
	if(debugOutput):		
		print(save_object)
	file_name=entry1.get()
	if file_name=='':
		print("No filename input, returning without saving")
		return
	with open(file_name, 'wb') as datafile:
		pickle.dump(save_object, datafile)
				
def loadFromDisk(): 									#check and read from disk "datafile"
	file_name=entry1.get()
	if file_name=='':
		print("No filename input, returning without loading")
	if os.path.isfile(file_name):
		with open(file_name, 'rb') as datafile:
			save_object = pickle.load(datafile)
	else:
		print("NO SAVE FILE FOUND")
		return
	if debugOutput:
		print(save_object)
	delAllExceptButtons()								#clear the current canvas
	counter_obj=save_object["counter_obj"]
	renderSaveData(save_object)							#renderAllObjects with older tags and set the counter to before
	
def delAllExceptButtons():								#deletes everything on screen except buttons
	print("Clearing screen")
	items = canvas.find_all()
	for curr_item in items:
		tags_list=canvas.itemcget(curr_item, "tags").split(" ")
		flag=False
		for curr_tag in tags_list:						#check all tags for button object
			if curr_tag.find("buttons")!=-1:
				flag=True
		if not flag:									#if its not a buttons object
			canvas.delete(curr_item)

def renderSaveData(save_object):						#renders data on screen based on passed object
	function_list={
		"freehand":cleanPolygon, 
		"line":getattr(canvas,"create_line"),
		"rectangle":getattr(canvas,"create_rectangle"),
		"oval":getattr(canvas,"create_oval"),
		"polygon":cleanPolygon
	}
	for drawing in save_object["objects"]:
		function_list[drawing["type"]](drawing["coords"], drawing["options"])
    
#MOUSE INPUT, FLOW MODIFYING FUNCTIONS
######################################

def left_click(event):
    global lastx, lasty, previous_object, polygon, previous_polygon, counter_obj, selection_id
    if mode == "draw":
        lastx, lasty = event.x, event.y
        previous_object=-1
        polygon=[]
        polygon.append((event.x,event.y))
        previous_polygon=-1
        if figure=="freehand":
            counter_obj["freehand"] = counter_obj["freehand"] + 1
    elif mode == "select":
        selection_id = findNearest(event)
        print(selection_id)
        if (selection_id!=-1):			#process further if left click closest is not a button
            selectOptions(event)
    elif mode == "advanced":
        selection_id = findNearest(event)
        print(selection_id)
        if (selection_id!=-1):			#process further if left click closest is not a button
            advancedOptions(event)
    else:
        print("ERROR:WRONG MODE DETECTED IN <left_click>")

def right_click(event):
	global lastx, lasty, previous_object, previous_polygon, polygon
	if mode=="draw":
		if figure == "polygon":
			delObject(previous_polygon)
			polygon.append((event.x,event.y))
			previous_polygon = drawPolygon(polygon, outline=color, width=2)
			lastx = event.x
			lasty = event.y
	elif mode=="select":
		if select_mode=="paste":
			renderClipboard(event)
	elif mode=="advanced":
		groupElements(event)
	else:
		print("ERROR:WRONG MODE DETECTED IN <right_click>")

def mouse_move(event):
	if mode=="draw": 
		if figure=="freehand":
			addFree(event);
		elif figure=="straight":
			addStraight(event);
		elif figure=="rectangle":
			addRectangle(event);
		elif figure=="ellipse":
			addEllipse(event);
		elif figure=="square":
			addSquare(event);
		elif figure=="circle":
			addCircle(event);
		elif figure=="polygon":
			addPolygon(event);
		else:
			print("ERROR IN CHOICE, CHECK CODE 2")
	elif mode=="select":
		if select_mode=="move":
			moveStuff(event)
		elif select_mode=="cut":
			print("cut")
		elif select_mode=="copy":
			print("copy")
		elif select_mode=="paste":
			print("paste")
		else:
			print("ERROR IN CHOICE, CHECK CODE 3")
	elif mode=="advanced":
		print("click drag in advanced mode detected")
	else:
		print("ERROR IN CHOICE, CHECK CODE 4")
	canvas.tag_raise("buttons") 									#to keep the objects with tag "buttons" on top, incase something is drawn over it



#VISUAL CONFIGURATION AND BINDING CODE STARTS
###############################

root = Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

canvas = Canvas(root)
canvas.grid(column=0, row=0, sticky=(N, W, E, S))
canvas.bind("<Button-1>", left_click)
canvas.bind("<B1-Motion>", mouse_move)
canvas.bind("<Button-2>", right_click)
canvas.bind("<Button-3>", right_click)

#COLOR BUTTONS RENDERING
avail_colors=["black", "red", "green", "blue", "cyan", "yellow", "magenta","white"]
counter=10
def color_lambda(var):
	return lambda x: setColor(var)
for curr in avail_colors:
	id = canvas.create_rectangle((10, counter, 30, counter+20), fill=curr, tags=('buttons'))
	canvas.tag_bind(id, "<Button-1>", color_lambda(curr))
	counter = counter + 25;

#DRAWING FIGURE BUTTONS RENDERING
avail_figures=["freehand", "straight", "rectangle", "ellipse", "square", "circle", "polygon"]
counter=35
def figure_lambda(var):
	return lambda x: setFigure(var)
for curr in avail_figures:
	id = canvas.create_rectangle((counter, 10, counter+70, 30), fill="white", tags=('buttons'))
	canvas.tag_bind(id, "<Button-1>", figure_lambda(curr))
	id = canvas.create_text((counter+5, 10), text=curr, tags=('buttons'), anchor=NW)
	canvas.tag_bind(id, "<Button-1>", figure_lambda(curr))
	counter = counter + 75;

#SELECTION BUTTONS RENDERING
selection_options=["move", "cut", "copy", "paste"]
#using previous counter value, to continue alongside figure buttons
def selection_lambda(var):
	return lambda x: setSelection(var)
for curr in selection_options:
	id = canvas.create_rectangle((counter, 10, counter+70, 30), fill="black", tags=('buttons'))
	canvas.tag_bind(id, "<Button-1>", selection_lambda(curr))
	id = canvas.create_text((counter+5, 10), text=curr, tags=('buttons'), anchor=NW, fill="white")
	canvas.tag_bind(id, "<Button-1>", selection_lambda(curr))
	counter = counter + 75;

#GROUPING/UNDO-REDO BUTTONS RENDERING
advanced_options=["group", "ungroup"]
#using previous counter value, to continue alongside figure buttons
def advanced_lambda(var):
	return lambda x: setAdvanced(var)
for curr in advanced_options:
	id = canvas.create_rectangle((counter, 10, counter+70, 30), fill="white", tags=('buttons'))
	canvas.tag_bind(id, "<Button-1>", advanced_lambda(curr))
	id = canvas.create_text((counter+5, 10), text=curr, tags=('buttons'), anchor=NW)
	canvas.tag_bind(id, "<Button-1>", advanced_lambda(curr))
	counter = counter + 75;

#DATA ENTRY BOX	
entry1 = Entry (root) 
canvas.create_window(counter, 10, window=entry1, anchor=NW, tags=('buttons'))
counter = counter + 170;

#SAVING/RELOADING BUTTONS RENDERING
saving_options=["save", "load"]
#using previous counter value, to continue alongside figure buttons
def saving_lambda(var):
	return lambda x: setSaved(var)
for curr in saving_options:
	id = canvas.create_rectangle((counter, 10, counter+70, 30), fill="black", tags=('buttons'))
	canvas.tag_bind(id, "<Button-1>", saving_lambda(curr))
	id = canvas.create_text((counter+5, 10), text=curr, tags=('buttons'), anchor=NW, fill="white")
	canvas.tag_bind(id, "<Button-1>", saving_lambda(curr))
	counter = counter + 75;

root.mainloop()
#MAIN CONFIGURATION CODE ENDS
#############################