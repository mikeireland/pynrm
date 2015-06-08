#Python code to generate a progress bar. This is the stand alone version for porting into projects and remove the need for a dependency on my personal module. JB

class progressbarClass:
    """Class to display a progress bar."""
    def __init__(self, finalcount):
        import sys
        self.finalcount=finalcount
        self.blockcount=0
        self.block="*"
        self.f=sys.stdout
        #
        # If the final count is zero, don't start the progress gauge
        #
        if not self.finalcount : return
        self.f.write('\n------------------ % Progress -------------------1\n')
        self.f.write('    1    2    3    4    5    6    7    8    9    0\n')
        self.f.write('----0----0----0----0----0----0----0----0----0----0\n')
        return

    def progress(self, count):
        if (count > self.finalcount): count=self.finalcount
        if (self.finalcount != 0) :
            percentcomplete=int(round(100*count/self.finalcount))

            if (percentcomplete < 1):
                percentcomplete=1
        else:
            percentcomplete=100

        blockcount=int(percentcomplete/2)
        if (blockcount > self.blockcount):
            for i in range(self.blockcount,blockcount):
                self.f.write(self.block)
                self.f.flush()

        if (percentcomplete == 100):
            self.f.write("\n")

        self.blockcount=blockcount
        return
