import numpy as np
import pbclass, os
try:
	import pyfits
except:
	import astropy.io.fits as pyfits
import tools, warnings
warnings.filterwarnings('ignore')

#A global
last_fnum = 0
last_block_string = ""

def write_textfile(g, block_string, name):
    global last_fnum
    global last_block_string
    if (last_block_string == block_string):
        return 
    current_fnum = int(name[name.find("n")+1:name.find(".fit")])
    numstr = "{0:04d} {1:02d} ".format(last_fnum, current_fnum-last_fnum)
    last_fnum = current_fnum
    if (last_block_string == ""):
        last_block_string=block_string
        return
    g.write(numstr+last_block_string + '\n')
    last_block_string=block_string

def popCSV(keys,operations,colheads,path,outfile,textfile='',blockkeys=[]):
    if ( (len(textfile)>0) & (len(blockkeys)>0) ):
        try:
            g=open(textfile,'w')
            g.write("FNUM NF ")
            for count, i in enumerate(blockkeys):
                if (count < 3):
                    g.write("{0:20s}".format(i))
                else:
                    g.write("{0:10s}".format(i))
            g.write("\n")
        except:
            print("Can not open text file for writing!")
            raise UserWarning
        textfile_open=True
    else:  
        print("No text file open - only creating CSV file")
        textfile_open=False
    with open(outfile,'w') as f:
        # write out the column headers to the file.
        f.write(",".join(colheads))
        
        # The fits header keywords stored to retrieve values from files

        # Use the walk command to step through all of the files in all directories starting at the "root" defined above
        for root, dirs, files in tools.sortedWalk(path):
            print 'Surveying directory ',root
            pb=pbclass.progressbarClass(np.size(files)-1)
            j=0
            for name in files:
                if "fits" in name:
                    pathAndName = os.path.join(root,name)
                    try:
                        prihdr = pyfits.getheader(pathAndName,ignore_missing_end=True) # get the primary header and values
                        image = pyfits.getdata(pathAndName,ignore_missing_end=True) # get the image data to be analysed
                    except Exception:
                        print 'Unable to open fits file'
                    values = [root, name] # The first two entries for the row
                    #extract values from header in keys
                    for i in keys:
                        try:
                            values.append(str(prihdr[i]))

                        except Exception:
                            values.append("")
                    
                    #Now for the "Block":
                    block_string = ""
                    for count,i in enumerate(blockkeys):
                        try:
                            if (count<3):
                                block_string += "{0:20s}".format(str(prihdr[i]))
                            else:
                                block_string += "{0:10s}".format(str(prihdr[i]))

                        except Exception:
                            print("Error with key" + i)


                    #start with manual operations
                    if prihdr["SAMPMODE"] == 2:
                        values.append("1")
                    else:
                        values.append(str(prihdr["MULTISAM"]))

                    # filtered version of the file used for peak and saturated
                    #filtered = tools.filter_image(pathAndName)
                    filtered = tools.filter_image(image)
                    # peak pixel
                    peakpix = tools.peak_coords(filtered)
                    values.append(str(peakpix[0])) # X coord of peak pixel
                    values.append(str(peakpix[1])) # Y coord of peak pixel
                    values.append(str(peakpix[2])) # value of peak pixel
                    # median pixel value in the image
                    values.append(str(np.median(image)))
                    # saturated
                    threshold = 20000 # max pixel value before considered saturated
                    #takes a numpy array, divides by the number of coadds and compares against threshold. returns true if peak pixel is above threshold
                    saturated= np.max(image/prihdr["coadds"]) > threshold
                
                    values.append(str(saturated))
                    
                    line = "\n" + ",".join(values)
                    f.write(line)
                    
                    #Now write our block text file...
                    if textfile_open:
                        write_textfile(g, block_string, name)
                
                j+=1
                pb.progress(j)
            write_textfile(g, block_string, name)
        return 1

