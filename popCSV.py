"""Notes:
This should additionally have the total integration time for the scan.

Also - a counter for the number fo visits in a night for 1 target.
"""

import numpy as np
import pbclass, os, matplotlib.pyplot as plt
try:
	import pyfits
except:
	import astropy.io.fits as pyfits
import tools, warnings
warnings.filterwarnings('ignore')

#A global
last_fnum = 0
last_block_string = ""
last_name=""
counter = 0
last_start = ""
def write_textfile(g, block_string, name,f_ix,fnum_prefix="",add_filename=False,total_int=None,instname='NIRC2'):
    global last_fnum
    global last_block_string
    global last_name
    global counter
    global last_start
    if instname=='NACO':
        newName = name
        hdr = pyfits.getheader(newName)
        name = hdr['ORIGFILE']
        nexp = hdr['ESO TPL NEXP']
        expno = hdr['ESO TPL EXPNO']
        start = hdr['ESO TPL START']
        counter+=1
        if start==last_start:
        	return
        last_counter = counter
        counter = 0
        #counter = nexp
        #print(name,last_counter)
        numstr = last_start+" {0:02d} ".format(last_counter)
        last_start = start
        if len(fnum_prefix) > 0 and fnum_prefix in name:
            current_fnum = int(name[name.find(fnum_prefix)+len(fnum_prefix):name.find(".fit")])
        else:
            current_fnum=f_ix
    elif instname=='NIRC2':
        counter+=1
        if (last_block_string == block_string):
            return 
        last_counter = counter
        counter = 0
        if len(fnum_prefix) > 0 and fnum_prefix in name:
            current_fnum = int(name[name.find(fnum_prefix)+len(fnum_prefix):name.find(".fit")])
        else:
            current_fnum=f_ix
        if last_fnum==0:
            last_fnum = current_fnum
        numstr = "{0:04d} {1:02d} ".format(last_fnum, last_counter)
    if (last_block_string == ""):
        last_block_string=block_string
        last_name=name
        return
    if add_filename:
        g.write(numstr+last_block_string + ' ' + last_name)    
    else:
        g.write(numstr+last_block_string)
    if total_int:
        g.write("{0:5.1f}".format(last_counter*total_int)+'\n')
    else:
        g.write('\n')
    last_fnum = current_fnum
    last_name = name
    last_block_string=block_string

def popCSV(keys,operations,colheads,path,outfile,textfile='',blockkeys=[],threshold=20000,fnum_prefix="n",add_filename=False,total_int_keys=None,instname='NIRC2'):
    """Populate a CSV file containing information about the fits headers
    
    Parameters
    ----------
    threshold: int
        The threshold before the file is considered saturated"""
    total_int=None
    if ( (len(textfile)>0) & (len(blockkeys)>0) ):
        try:
            g=open(textfile,'w')
            allFiles = os.listdir(path)
            if instname=='NACO':
                g.write("START               NF ")
            elif instname=='NIRC2':
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
            print('Surveying directory ',root)
            pb=pbclass.progressbarClass(np.size(files)-1)
            j=0
            for f_ix,name in enumerate(files):
                if "fit" in name:
                    pathAndName = os.path.join(root,name)
                    try:
                        prihdr = pyfits.getheader(pathAndName,ignore_missing_end=True) # get the primary header and values
                        image = pyfits.getdata(pathAndName,ignore_missing_end=True) # get the image data to be analysed
                    except Exception:
                        print('Unable to open fits file')
                    values = [root, name] # The first two entries for the row
                    if instname=='NACO' and 'ORIGFILE' in prihdr.keys():
                    	oldName = prihdr['ORIGFILE']
                    	fnum_prefix = oldName[0:oldName.find('.fit')-4]
                    	#name = oldName
                    #extract values from header in keys
                    if instname=='NACO':
                        for i in keys:
                            if i in prihdr:
                                if i=='OBJECT':
                                    if prihdr[i] == 'Object name not set':
                                        if 'Flat' in prihdr['ESO TPL ID']:
                                            values.append('flats')
                                        elif 'Dark' in prihdr['ESO TPL ID']:
                                            values.append('darks')
                                        else:
                                            values.append("")
                                    else:
                                        values.append(str(prihdr[i]))
                                else:
                                    values.append(str(prihdr[i]))
                            else:
                                if i=='SHRNAME':
                                    if 'Dark' in prihdr['ESO TPL ID']:
                                        values.append('closed')
                                    else:
                                        values.append('open')
                                elif i=='SLITNAME':
                                    values.append('none')
                                elif i=='ESO TEL ALT':
                                    values.append('45')
                                elif i=='COADDS':
                                    values.append('1')
                                else:
                                    values.append("")
                    elif instname=='NIRC2':
                        for i in keys:
                            try:
                                values.append(str(prihdr[i]))

                            except Exception:
                                values.append("")
                    
                    #Now for the "Block":
                    block_string = ""
                    for count,i in enumerate(blockkeys):
                        try:		
                            if i in prihdr:
                                if i=='OBJECT' and instname=='NACO':
                                    if prihdr[i] == 'Object name not set':
                                        if 'Flat' in prihdr['ESO TPL ID']:
                                            objName = 'flats'
                                        elif 'Dark' in prihdr['ESO TPL ID']:
                                            objName = 'darks'
                                    else:
                                        objName = prihdr[i]
                                    if (count<3):
                                        block_string += "{0:20s}".format(objName)
                                    else:
                                        block_string += "{0:10s}".format(objName)
                                else:
                                    if (count<3):
                                        block_string += "{0:20s}".format(str(prihdr[i]))
                                    else:
                                        block_string += "{0:10s}".format(str(prihdr[i]))
                            else:
                                if i=='COADDS':
                                    string = '1'
                                else:
                                    string = 'unknown'
                                if (count<3):
                                    block_string += "{0:20s}".format(string)
                                else:
                                    block_string += "{0:10s}".format(string)

                        except Exception:
                            print("Error with key" + i)


                    #start with manual operations specific to telescopes
                    if ("CURRINST" in prihdr.keys() and prihdr["CURRINST"] == "NIRC2"):
                        if prihdr["SAMPMODE"] == 2:
                            values.append("1")
                        else:
                            values.append(str(prihdr["MULTISAM"]))
                    else:
                       values.append("")
                    # filtered version of the file used for peak and saturated
                    #filtered = tools.filter_image(pathAndName)
                    if len(image.shape)==2:
                        filtered = tools.filter_image(image)
                    else:
                        filtered = np.median(image,axis=0)
                    # peak pixel
                    peakpix = tools.peak_coords(filtered)
                    values.append(str(peakpix[0])) # X coord of peak pixel
                    values.append(str(peakpix[1])) # Y coord of peak pixel
                    values.append(str(peakpix[2])) # value of peak pixel
                    # median pixel value in the image
                    values.append(str(np.median(image)))
                    # saturated
                    #takes a numpy array, divides by the number of coadds and compares against threshold. returns true if peak pixel is above threshold
                    if "COADDS" in prihdr.keys():
                        saturated= np.max(image/prihdr["COADDS"]) > threshold
                    else:
                        saturated=np.max(image) > threshold
                    values.append(str(saturated))
                    
                    line = "\n" + ",".join(values)
                    f.write(line)
                    
                    #Now write our block text file...
                    if total_int_keys:
                        total_int=1
                        for akey in total_int_keys:
                            if akey in prihdr.keys():
                                total_int *= prihdr[akey]
                    if textfile_open:
                        write_textfile(g, block_string, name,f_ix,fnum_prefix=fnum_prefix,add_filename=add_filename,total_int=total_int,instname=instname)
                
                j+=1
                pb.progress(j)
            if "fit" in name:
                write_textfile(g, "", name,f_ix,fnum_prefix=fnum_prefix,add_filename=add_filename,total_int=total_int,instname=instname)
        return 1

