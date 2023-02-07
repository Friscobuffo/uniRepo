from time import time as ltime0316

class ProgressBar():
    __fill = '█'

    def __init__(self, iterations):
        self.__iterations = iterations
        self.__counter = 0
        self.__start = ltime0316()
        self.__bar_length = 39
    
    def __fill_fun(self, val):
        if val==0.0: return '█'
        if val<0.125: return '▏'
        if val<0.25: return '▎'
        if val<0.375: return '▍'
        if val<0.5: return '▌'
        if val<0.625: return '▋'
        if val<0.75: return '▊'
        if val<0.875: return '▉'
        return '█'

    def __update(self):
        totTime = ltime0316() - self.__start
        freq = self.__freq(self.__counter / totTime)
        eta = self.__eta(freq)
        if type(freq) == int:
            frequency = str(freq)
        else:
            frequency = "{:.2f}".format(freq)
        percent = 100*(self.__counter / float(self.__iterations))
        spercent = "{:.2f}".format(percent)
        filledLength = self.__bar_length * self.__counter // self.__iterations
        factor = (self.__bar_length * self.__counter / self.__iterations) - (self.__bar_length * self.__counter // self.__iterations)
        bar = '|' + (self.__fill * (filledLength)) + self.__fill_fun(factor) + ' ' * (self.__bar_length - filledLength) + '|'
        print(f'\r{bar} [{spercent}]% (eta: {eta}s, frequency: {frequency} frame per second)'+7*" ", end = '', flush = True)

    def next(self):
        self.__counter += 1
        if self.__counter == self.__iterations:
            self.__finished()
        else:
            self.__update()

    def __finished(self):
        bar = '|' + (self.__fill * (self.__bar_length+1)) + '|'
        totTime = ltime0316() - self.__start
        if totTime >= 60:
            mins = int(totTime/60)
            secs = int(totTime%60)
            totalTime = str(mins) +" mins " + str(secs) + " seconds"
        else:
            totalTime = "{:.2f}".format(totTime) + " seconds"
        print(f'\r{bar} [100.00]% (total time: {totalTime})'+(20)*" ")
    
    def __freq(self, freq):
        if freq > 10: return int(freq)
        return round(freq, 2)

    def __eta(self, freq):
        eta = int((self.__iterations-self.__counter) / freq)
        mins = ""
        if eta > 60:
            mins = int(eta/60)
            mins = str(mins) + "mins "
            eta = eta%60
        return mins + str(eta)