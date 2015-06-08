#!/usr/bin/python

# preliminary version to build a CSV with relevant information mainly from fits headers
# "saturated" will be added as a function using median pixel and dividing by coadds
# then checking for the peak value being below some threshold (20,000?)


#import sort_tools
import csv, jpb

path = jpb.ask('Which directory to run this code?','./') # path to root of directory to walk through, relative to location of script
infile = path + "datainfo.csv" # the input file to read from
outfile = path + "filteredCSV.csv" # the output file to write to

#column headers as of 31/12/12 in popCSV
#0  "DIRECTORY"
#1  "FILENAME"
#2  "FILTER"
#3  "OBJECT"
#4  "TARGNAME"
#5  "CAMNAME"
#6  "COADDS"
#7  "ITIME"
#8  "DATE-OBS"
#9  "MJD-OBS"
#10 "UTC"
#11 "EL"
#12 "NAXIS1"
#13 "NAXIS2"
#14 "PIXSCALE"
#15 "SLITNAME"
#16 "MULTISAM"
#17 "PEAKPIX(X)"
#18 "PEAKPIX(Y)"
#19 "PEAK_VALUE"
#20 "MEDIAN_VALUE"
#21 "SATURATED"

#indices of relevant keys
saturated = 21
peak_value = 19
median = 20
object_key = 3
juls_key = 9
filter_key = 2
camera_key = 5

dark_limit = 1000 #threshold for dark -> peak < threshold -> dark
med_peak = 0.85 #threshold for median/peak ratio
juls_hour = 0.04 #length of hour in julian time


file_contents = [] #empty list ready to receive the goods

with open(infile,'rb') as csvfile:
    file_reader = csv.reader(csvfile, delimiter=',')
    #file_contents.append(file_reader.next()) #puts the first line into the list and moves to the next line of the csv.reader object\\
    headers = file_reader.next() #skip the first row -> column headers (save headers for later)
    for row in file_reader: #step through each line
        if row[saturated] == 'False' and float(row[peak_value]) > dark_limit and \
                (float(row[median])/float(row[peak_value])) < med_peak:
            file_contents.append(row)


#######
#now that there is a list of lists of potentials, need to sort
#sorted in this order ===> date -> camera > filter -> object
#first -> last
#sorted_values = sorted(sorted(sorted(sorted(file_contents, key = lambda date: date[juls_key]), key = lambda filt: filt[filter_key]), \
#    key = lambda cam: cam[camera_key]), key = lambda obj: obj[object_key])
sorted_values = sorted(sorted(sorted(sorted(file_contents, key = lambda date: date[juls_key]), key = lambda cam: cam[camera_key]), \
    key = lambda filt: filt[filter_key]), key = lambda obj: obj[object_key])

#######
#now that they're sorted, need to check to see if they're within an hour of each other
#if they have the same object name
#if there is less than 2 observations in the same hour, don't include them.
#must have the same filter and camera used for obs

#for value in sorted_values:
#    if value[object_key] == value[object_key + 1] and value[filter_key] == value[filter_key + 1] and \
#            value[camera_key] == value[camera_key + 1] and #time check
#        #include this in the final
#    else if 


# master_list will hold the "master" value to check against when comparing values
# at the end of each iteration of the for loop below (in the with), the next value in
# sorted_values will overwrite the value in master_list.
master_list = [sorted_values[0]]
obj_list = master_list
prev_obj = ""
obj_full = False

with open(outfile,'wb') as f:
    writer = csv.writer(f)
    writer.writerow(headers) #write the headers to the first line (headers assigned in previous "with")
    #start from the second value in the list
    #because comparing to the first initially
    sorted_length = len(sorted_values) - 1
    sorted_iterator = 1 #starting with second value
    for value in sorted_values[1:]:
        if obj_full:
            obj_list = []
        obj_full = False
        sorted_iterator += 1 # used to see if we're at the end of the list
        #if obj_list[0][object_key] == value[object_key]: #are they the same object?
        if master_list[0][object_key] == value[object_key]: #are they the same object?
            obj_list.append(value)
            #sorted_iterator += 1
            #print "just appended an object"
            #print value[object_key]
            # if we're at the end of the list, then no more possible objects, hence "full"
            if sorted_iterator == sorted_length:
                obj_full = True
        else:
            obj_full = True
        #print obj_full == True and len(obj_list) > 1
        #print "obj_full == True is: ", obj_full == True 
        #print "len(obj_list) > 1 is: ", len(obj_list) > 1
        #print "-----------------"

        if obj_full == True and len(obj_list) > 1: #now obj_list is a full list of all the same object
            #do something with the list.....
            print "THE LENGTH OF OBJ_LIST IS: ", len(obj_list)
            master_filt = [obj_list[0]]
            filt_list = master_filt
            #now making a list with the same filter
            #print "starting on the same filter part"
            obj_length = len(obj_list) - 1
            obj_iterator = 1 #starting with second value
            filt_full = False #initialise it
            for obj in obj_list[1:]:
                if filt_full:
                    filt_list = []
                filt_full = False
                obj_iterator += 1
                if master_filt[0][filter_key] == obj[filter_key]: #are they the same filter?
                    filt_list.append(obj)
                    #obj_iterator += 1
                    #print "just appended another filter"
                    #check to see if full list
                    if obj_iterator == obj_length:
                        filt_full = True
                    #do stuff
                else:
                    filt_full = True
                print "filt_full == True is: ", filt_full == True 
                print "len(filt_list) > 1 is: ", len(filt_list) > 1
                print "value for obj[juls_key] is: ", obj[juls_key]
                print "-----------------"

                if filt_full and len(filt_list) > 1: #should be full list of same obj + filter
                    #print "THE LENGTH OF filt_list IS: ", len(filt_list)
                    #do something
                    master_cam = [filt_list[0]]
                    cam_list = master_cam
                    #print "have full list with more than 1 filter+obj"
                    #now making a list with the same camera

                    filt_length = len(filt_list) - 1
                    filt_iterator = 1 #starting with second value
                    cam_full = False #initialise it
                    for filt in filt_list[1:]:
                        if cam_full:
                            cam_list = []
                        cam_full = False
                        filt_iterator += 1
                        #print "starting same camera part"
                        #print cam_list[0][camera_key]
                        if master_cam[0][camera_key] == filt[camera_key]: #are they the same camera?
                            cam_list.append(filt)
                            filt_iterator += 1
                            #print "just appended another camera"
                            #print len(cam_list)
                            #print "****************************************"
                            if filt_iterator == filt_length:
                                cam_full = True
                        else:
                            cam_full = True
                        if cam_full and len(cam_list) > 1: #full list of all the same and more than 1
                            #now to compare the times
                            #1hr in julian time is ~0.04
                            #juls_hour = 0.04
                            #print "have full list with more than 1 filter+obj+cam"
                            #time_list = cam_list[0]
                            master_time = cam_list[0]
                            time_list = [master_time]
                            #i = 0
                            time_iterator = 1
                            for cam in cam_list[1:]:
                                #check julian date (juls_key)
                                #print "##############################################"
                                #print float(cam[juls_key]) - float(time_list[i][juls_key])
                                #if float(cam[juls_key]) - float(time_list[i][juls_key]) < juls_hour:
                                #if float(cam[juls_key]) - float(time_list[juls_key]) < juls_hour:
                                time_full = False
                                time_iterator += 1
                                if float(cam[juls_key]) - float(master_time[juls_key]) < juls_hour:
                                    time_list.append(cam)
                                    #print "the dates were within 1 hour of each other, appended"
                                    if time_iterator == len(cam_list):
                                        time_full = True
                                else:
                                    #print "else -> time_list"
                                    master_time = [cam]
                                    time_full = True
                                if len(time_list) > 1 and time_full: #everything filtered and within same hour of the next
                                    #print everything in time_list to file
                                    #print "more than 1 item within the same hour, print them to the file!"
                                    writer.writerows(time_list)
                                    #reset time_list
                                    master_time = [cam]
                            #cam_list = [filt]
                            master_cam = [filt]
                        else:
                            #print "else -> cam_list"
                            #cam_list = [filt]
                            master_cam = [filt]
                    #filt_list = [obj]
                    master_filt = [obj]
                else:
                    #print "else -> filt_list"
                    #filt_list = [obj]
                    master_filt = [obj]

            #go to the next value in the list of sorted values. 
            #obj_list = [value]
            master_list = [value]


        else: #there was only a single object in the list, ignore and go to next
            #obj_list = [value]
            master_list = [value]



