# -*- coding: utf-8 -*-
"""
Base Version from Mon Oct 30 13:33:36 2023

@author: Seriosha Remmlinger
"""

#Version 5C

from scipy import signal
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os



#edit here to add a HOME directory, etc.


_FILE = 'wavefile.wav'

info_line = 'PLOT TITLE'

Hz_per_tick = 3
sec_per_rev = 1.8  # 1.8 for 33, 1.33 for 45, .766 for 78.26

stereo_channel = 0    #0 = Left, 1 = Right
filter_freq = 60

#end edit



def instfreq(sig,Fs,filter_freq):
    z = signal.hilbert(sig)
    rawfreq = Fs/(2*np.pi)*np.diff(np.unwrap(np.angle(z)))
    rawfreq = np.append(rawfreq,rawfreq[len(rawfreq)-1])    #np.diff drops one end point

    b, a = signal.iirfilter(1,filter_freq/(Fs/2), btype='lowpass')
    zi = signal.lfilter_zi(b, a) #Initialize the filter to the mean of the leading edge of the data
    rawfreq,_ = signal.lfilter(b,a,rawfreq,zi=zi*np.mean(rawfreq[0:2000])) #reduces glitch, first pass

    b, a = signal.iirfilter(3,filter_freq/(Fs/2), btype='lowpass')
    instfreq = signal.filtfilt(b,a,rawfreq) #back and forth linear phase IIR filter (6 pole)

    return (instfreq)


y = read(_FILE)
Fs = float(y[0])
if np.size(y[1][0]) == 2:
    sig = y[1][:,stereo_channel][0:int(Fs*(sec_per_rev*3))] #Grab 3*sec_per_rev of audio from the specified channel
else:
    sig = y[1][0:int(Fs*(sec_per_rev*3))] #mono file

t = np.arange(sec_per_rev,0,-1/Fs)  #Reverse time (theta axis)
theta = t*2*np.pi/sec_per_rev   #Time becomes degrees (1 rev = 2pi radians)
theta = np.roll(theta,int((sec_per_rev*Fs/4)))  #Rotate 90 deg to put 0 on top (sec_per_rev*Fs/4)

freq1 = instfreq(sig,Fs,filter_freq)

freq1 = np.roll(freq1,-int(Fs*.2))#Throw away the first .2sec to guarantee the IIR transient settles
if1 = freq1[0:int(Fs*sec_per_rev)]
if2 = freq1[int(Fs*sec_per_rev):int(2*Fs*sec_per_rev)]


maxf = (max(max(if1),max(if2))+.2) #This shuld be changed to make the maximum frequency an even 10th of a Hz.

r1 = 20.-(maxf-if1)/Hz_per_tick  #20 radial ticks at Hz_per_tick is fixed, adaptive scaling
r2 = 20.-(maxf-if2)/Hz_per_tick  #is an exercise for later

#plt.figure(1)
plt.figure(figsize=(11,11))
ax = plt.subplot(111, projection='polar')
ax.plot(theta,r1)
ax.plot(theta,r2)

dgr = (2*np.pi)/360.


mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(_FILE))
ax.text(226.*dgr, 28.5, 'Mean Rev1 {:4.3f}Hz'.format(np.mean(if1)) + "\n" + \
    'Mean Rev2 {:4.3f}Hz'.format(np.mean(if2)) + "\n" + \
    _FILE + "\n" + \
    mod_date.strftime("%b %d, %Y %H:%M:%S"), fontsize=9)


ax.set_rmax(20)
                #Set up the ticks y is radial x is theta, it turns out x and y
                #methods still work in polar projection but sometimes do funny things

tick_loc = np.arange(1,21,1)


myticks = []
for x in range(0,20,1):
    myticks.append('{:4.2f}Hz'.format(maxf-(19*Hz_per_tick)+x*Hz_per_tick))

ax.set_rgrids(tick_loc, labels = myticks, angle = 90, fontsize = 8)

ax.set_xticklabels(['90'+u'\N{DEGREE SIGN}','45'+u'\N{DEGREE SIGN}','0'+u'\N{DEGREE SIGN}',\
                    '315'+u'\N{DEGREE SIGN}','270'+u'\N{DEGREE SIGN}','225'+u'\N{DEGREE SIGN}',\
                    '180'+u'\N{DEGREE SIGN}','135'+u'\N{DEGREE SIGN}'])



ax.grid(True)

ax.set_title(info_line, va='bottom', fontsize=16)

plt.savefig(info_line.replace(' / ', '_') +'.png', bbox_inches='tight', pad_inches=.5)

plt.show()