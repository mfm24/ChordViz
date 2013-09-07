# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:09:10 2013

@author: matt
# based on MyPlayer3_Callback (which is newer than MyPlayer3.py)
"""
from __future__ import division

import time, math, logging
import numpy as np
from threading import Lock, Thread
import itertools

# not sure I've added correct path in launchd.conf
# and export doesn't obviously work
import sys
sys.path.append('/Users/matt/Dropbox/personal/dev/PythonLibs/')
try:
    from uidecorators import ui_decorators
    use_ui = True
except ImportError:
    # a bit nasty. We'll create an object were all members
    # return a decorator function returning a decorator that does nothing!
    class FakeUIDec:
        def __getattr__(self, name):
            def no_wrap(*args, **kwargs):
                def wrap_creator(func):
                    def w(*args, **kwargs):
                        func(*args, **kwargs)
                    return w
                return wrap_creator
            return no_wrap
            
    ui_decorators = FakeUIDec()
    use_ui=False
    

try:
    import pyaudio
    p = pyaudio.PyAudio()
    has_pyaudio = True
except ImportError:
    logging.warn("PyAudio not found! - Will not be able to output any audio!")
    has_pyaudio = False


def play_waveform(w):
    def callback(in_data, frame_count, time_info, status):
        # this requests upto 1024 frames?
        with w.datalock:
            ndata = w.data
        if ndata is not None:
            return (np.hstack([ndata]*(frame_count//1024)), pyaudio.paContinue)
        else:
            return (None, pyaudio.paComplete)
    if has_pyaudio:
        # open stream using callback (3)
        play_waveform.stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=w.rate,
                        output=True,
                        frames_per_buffer=w.size,
                        stream_callback=callback)

                    
play_waveform.stream = None
                
note_types = {
    "PureTone": lambda harmonic: 1 if harmonic==0 else 0,
    "Poisson0.5": lambda harmonic: poisson(0.5, harmonic),
    "Poisson1": lambda harmonic: poisson(1, harmonic),
    "Poisson2": lambda harmonic: poisson(2, harmonic),
    "Poisson3": lambda harmonic: poisson(3, harmonic),
    "Lorentz1": lambda harmonic: 1.0/(1.0+harmonic**2),
    "Lorentz10": lambda harmonic: 10.0/(10.0+harmonic**2),
    "Equal": lambda harmonic: 1,
    "EqualOdd": lambda harmonic: 1 if harmonic%2==1 or harmonic==0 else 0,
    "EqualEven": lambda harmonic: 1 if harmonic%2==0 else 0,
    "OneOverX": lambda harmonic: 1.0/(harmonic+1.0)
    }

equal_temperament_notes = [2 ** (x / 12.0) for x in range(12)]
just_intonation_notes = [1, 16 / 15., 9 / 8., 6 / 5., 5 / 4., 4 / 3., 45 / 32., 3 / 2., 8 / 5., 5 / 3., 16 / 9., 15 / 8.]
twelve_tone_names = ["I", "IIb", "II", "IIIb", "III", "IV", "IV#", "V", "VIb", "VI", "VIIb", "VII"]
class Waveform(object):    
    def __init__(self, size=1024*16, rate=44100):
        self.size = size
        self.rate = rate
        self.data = np.zeros((size), dtype=np.int16)
        self.datalock = Lock()
        self.volume_amp = 0.1
        self.form = lambda note: poisson(2, note)
        self.notetype="Poisson1"
        self.notefreq=440
        self.on_notes_changed=[]
        self._harmonics_slice = None
        self.clear_notes()

    def clear_notes(self):
        self.notes = []
        self()
            
    def set_notes(self, notes):
        self.clear_notes()
        self.add_notes(notes)
        self()
            
    def add_notes(self, notes):
        self.notes.append(list(notes))
        self()
                   
    def __call__(self):
        newdata = np.zeros((self.size), dtype=np.complex64)
        for notegroup in self.notes:
            for freq, mag in notegroup:
                dphase=int (freq*self.size / self.rate )
                logging.info("Adding note at pixel %s", dphase)
                if dphase > len(newdata)/2:
                    continue  # this is nyquist, can't go any higher
                #let's scale mag by number of notes 
                newdata[dphase]=self.volume_amp*mag*32765/2
                #make ft real
                newdata[-dphase] = np.conj(newdata[dphase])
        sqrtsumsq = math.sqrt((newdata**2).sum())
        if sqrtsumsq:
            newdata *= self.volume_amp * 2.0 * 32767.0 / sqrtsumsq
        printimag = 0
        if printimag:
            complex_d=np.imag(np.fft.fft(newdata));
            print "imag magnitude: ", np.sqrt(np.sum(complex_d**2))
        newdata = np.asarray(np.real(np.fft.fft(newdata)), dtype=np.int16)
        with self.datalock:
            self.data = newdata
        for f in self.on_notes_changed:
            f()
       
    def get_volume(self):
        v = math.log(self.volume_amp, 10)*20
        return v
        
    @ui_decorators.slider(getfunc=get_volume, maximum=0, minimum=-50, scale=1)     
    def volume(self, value):
        self.volume_amp = 10**(value/20.0)
        self()
        
    def get_note_type(self):
        return self.notetype
        
    @ui_decorators.combobox(
        getfunc=get_note_type, 
        options=note_types.keys())      
    def note_type(self, t):
        self.notetype = t
       
    def get_harmonics_slice(self):
        if self._harmonics_slice:
            return ",".join(self._harmonics_slice)
        else:
            return ""
        
    @ui_decorators.textbox(getfunc=get_harmonics_slice)
    def harmonics_slice(self, n):
        """
        Sets the harmonics to display
        Should be either [start:]stop[:step]
        or else a,b,c where a,b,c are indices to choose
        """
        if n=="":
            self._harmonics_slice = None
            return
        if ':' in n:
            sc = [int(x or "0") for x in n.split(":")]
            if len(sc)==1:
                self._harmonics_slice = (None, sc[0], None)
            elif len(sc) == 2:
                self._harmonics_slice = (sc[0], sc[1], None)
            else:
                self._harmonics_slice = (sc[0], sc[1], sc[2])
        else:
            self._harmonics_slice = [int(x or "-1") for x in n.split(',')]

        
    def get_root_frequency(self):
        return self.notefreq
        
    @ui_decorators.textbox(getfunc=get_root_frequency)
    def root_frequency(self, val):
        self.notefreq = float(val)
        self()
        
    def add_form(self, root):
        if isinstance(self._harmonics_slice, list):
            all_notes = list(notes_from_func(note_types[self.notetype], root))
            notes = []
            for i in self._harmonics_slice:
                notes.append(all_notes[i])
        else:
            slice_args = self._harmonics_slice or (None,)
            notes = itertools.islice(
                notes_from_func(note_types[self.notetype], root),
                *slice_args)
        self.add_notes(notes)
    
    @ui_decorators.button
    def clear(self):
        self.clear_notes()
    
    @ui_decorators.button
    def note_root(self):
        self.add_form(self.notefreq)
        self()
    @ui_decorators.button
    def note_major3rd(self):
        self.add_form(self.notefreq*5.0/4.0)
        self()
        
    @ui_decorators.button
    def note_fifth(self):
        self.add_form(self.notefreq*6.0/4.0)
        self()        
        
    @ui_decorators.button
    def play_major_chord(self):
        self.play_threaded_chord([self.notefreq,
                             self.notefreq*5.0/4.0,
                             self.notefreq*6.0/4.0])
 
    @ui_decorators.button
    def test(self):
        self.play_threaded_chord([self.notefreq,
                             self.notefreq*7.0/8.0,
                             self.notefreq*6.0/4.0])   
    @ui_decorators.button
    def play_minor_chord(self):
        self.play_threaded_chord([self.notefreq,
                             self.notefreq*12.0/10.0,
                             self.notefreq*15.0/10.0])
                             
    @ui_decorators.button
    def play_minor_chord_fifth(self):
        self.play_threaded_chord([self.notefreq,
                             self.notefreq*4.0/3.0,
                             self.notefreq*8.0/5.0])
                                             
        
    def play_threaded_chord(self, roots):
        def run_through():
            for i,n in enumerate(roots):
                self.clear_notes()
                [self.add_form([]) for t in range(i)]
                self.add_form(n)
                time.sleep(1.5)
            self.clear_notes()
            for n in roots:
                self.add_form(n)
        Thread(target=run_through).start()
        
# run in interactive shell and use set_notes to play?
def poisson(l, n):
    return math.exp(-l)*l**n/math.factorial(n)
    
def notes_from_func(func, root):
    for h in itertools.count():
        mag = func(h)
        # we cut off until we reach 22.1kHz
        if root+root*h > 22100:
            return
        yield root+root*h, mag
        
def cleanup():
    if has_pyaudio:
        play_waveform.stream.close()
        p.terminate()

######################## UI Stuff ############################
# this could go in a separate file, but keeping it here for the 
# moment

# creating a UI Options class for modifying the visualisation using
# out qt decorators
class UIOptions:
    def __init__(self):
        self._linear_freq_in_octaves = True
        self.virtual_size = 1500,1500
        self._inverse = True
        self._show_just_notes = True
        self._show_base_spiral = True
        self._show_ET_notes = False  # ET=equal temperament
        
    def get_linear_freq_in_octaves(self):
        return self._linear_freq_in_octaves
        
    @ui_decorators.checkbox(getfunc=get_linear_freq_in_octaves)
    def linear_freq_in_octaves(self, newval):
        self._linear_freq_in_octaves = newval
        notes_changed()
        
    def get_show_base_spiral(self):
        return self._show_base_spiral
        
    @ui_decorators.checkbox(getfunc=get_show_base_spiral)
    def show_base_spiral(self, newval):
        self._show_base_spiral = newval
        notes_changed()
        
    def get_inverse(self):
        return self._inverse
    
    @ui_decorators.checkbox(getfunc=get_inverse)
    def inverse(self, newval):
        self._inverse = newval
        notes_changed()
        
    def get_show_just_notes(self):
        return self._show_just_notes
    
    @ui_decorators.checkbox(getfunc=get_show_just_notes)
    def show_just_notes(self, newval):
        self._show_just_notes = newval
        notes_changed()
        
    def get_show_ET_notes(self):
        return self._show_ET_notes
    
    @ui_decorators.checkbox(getfunc=get_show_ET_notes)
    def show_ET_notes(self, newval):
        self._show_ET_notes = newval
        notes_changed()
        

def make_note_lines(root, named_notes, width, radius):
    """
    For the dictionary named_notes, draws thin lines for each note
    adding the key for the note to the SVG.
    This way we can overlay scales on the diagrams.
    """
    lines = []
    for name, freq in named_notes.iteritems():
        (x1, y1), theta = get_pos_theta_for_note(freq, root, 0, 0)
        font_size = radius/16.0
        lines.append(
            '<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="{width}"/>'.format(
            x1=x1, x2=x1 + 2 * radius * math.sin(theta),
            y1=y1, y2=y1 - 2 * radius * math.cos(theta),
            width=width))
        lines.append('<text x="{x}" y="{y}" font-size="{fs}">{text}</text>'.format(
            x=x1 + radius * math.sin(theta),
            y=y1 - radius * math.cos(theta),
            text=name, fs=font_size))
    return "\n".join(lines)

def get_pos_theta_for_note(f, root, root_radius, length):
    """
    Return (x,y),theta where (x,y) is the starting position of the note
    and theta is the angle the note should have
    """
    # first, we calculate the octave and theta for the root
    logf = math.log(f / root, 2)
    note_fraction, octave = math.modf(logf)
    if ui_opts.get_linear_freq_in_octaves():
        note = (2**note_fraction - 1)
    else:
        note = note_fraction
    theta = note * 2.0 * math.pi
    
    centerx, centery = (x / 2 for x in ui_opts.virtual_size)
    r = root_radius + (octave + note_fraction) * length
    x = centerx + r * math.sin(theta)
    y = centery - r * math.cos(theta)
    return (x,y), theta
    
def make_spiral_lines_from_notes(root, notes, length=75, root_radius=100):
    """
    Is there a way to represent notes where octaves are still seperated but
    we can see notes of the same pitch?
    We could draw a spiral, where an octave is 360 degrees and on the next
    ring out.
    There's a similar idea here:
    http://nastechservices.com/Spectrograms.html
    How should we represent a 3:2 ratio? If wejust take log(x,2)*2*pi
    then 3/2 is at 210deg (or 3.67rad). Is it worth making the scale linear,
    and putting 3/2 at 180deg? We could also spiral so that 3/2f gets us to 180
    deg then we stretch out the remaining part of the curve?
    We'll try the linear for now.
    It works, but not all 3/2 notes are 180deg from each other
    (if the higher note is past the root, it's not)
    Is there a way to do this? Maybe not, eg we make 5th = 3r/2 opposite root
    and 3/2r = 9/4 != root and yet root still needs to be 180deg from it
    """
    stroke_width_scale = 15
    width_gamma = 0.2  # we use width^this as the width 
    centerx, centery = (x / 2 for x in ui_opts.virtual_size)
    lines = []
    for f, m in notes:
        # we split the note into octave and note (0 - 1)
        width = stroke_width_scale * math.pow(m, width_gamma)

        (x1, y1), theta = get_pos_theta_for_note(f, root, root_radius, length)

        x2 = x1 + 0.9 * length * math.sin(theta)
        y2 = y1 - 0.9 * length * math.cos(theta)

        lines.append('<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="{width}"/>'.format(
                     x1=x1, x2=x2, y1=y1, y2=y2,
                     width=width))
    return "\n".join(lines)
    
def make_spiral_octave_lines(root, length=75, root_radius=100, max_f=22100):
    """
    Starting with the root note, draw the spiral on which
    any higher frequency notes will sit. This way we can count
    harmonics more easily
    """
    width = 0.5 
    (x1, y1), _ =  get_pos_theta_for_note(root, root, root_radius, length)
    lines = []
    step = int(root/50) or 1
    for f in range(int(root), int(max_f), step):
        (x2, y2), theta = get_pos_theta_for_note(f, root, root_radius, length)
        lines.append('<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="{width}"/>'.format(
             x1=x1, x2=x2, y1=y1, y2=y2,
             width=width))
        x1, y1 = x2, y2
    return "\n".join(lines)
            

rgb_colors = [0xFF0000, 0x00FF00, 0x0000FF]
cym_colors = [0x00FFFF, 0xFF00FF, 0xFFFF00]
white = 0xFFFFFFFF
black = 0xFF000000

# some QT specific stuff follows:
import PySide.QtCore
import PySide.QtGui
import PySide.QtSvg


def render_svg(svg, qp):
    r = PySide.QtSvg.QSvgRenderer()
    w,h = ui_opts.virtual_size
    ret = '<svg  xmlns="http://www.w3.org/2000/svg" version="1.1" width="{}" height="{}">'.format(w, h)
    ret += svg
    ret += "</svg>"
    # print ret
    r.load(PySide.QtCore.QByteArray(ret))
    assert r.isValid()
    r.render(qp)

def raw_svg_to_group(svg, color, extras=""):
    ret = '<g stroke="#{0:06X}" fill="#{0:06X}" {1}>'.format(
        color & 0xFFFFFF, extras)
    ret += svg
    ret += "</g>"
    return ret
    
from uidecorators.qt_framework import Framework

def notes_changed(*args):
    mode = "inverse" if ui_opts.get_inverse() else "normal"
    qim = PySide.QtGui.QImage(d.widget().width(), d.widget().height(), PySide.QtGui.QImage.Format.Format_ARGB32)
    qp = PySide.QtGui.QPainter(qim)
    qp.setRenderHint(qp.Antialiasing)
    qp.setRenderHint(qp.SmoothPixmapTransform)
    
    if mode == "inverse":
        #qim.fill(white)
        qp.setCompositionMode(qp.CompositionMode.CompositionMode_Darken)
        colors = cym_colors
        default_foreground = black
        default_background = white
        mode = "darken"
    else:
        #qim.fill(black)
        qp.setCompositionMode(qp.CompositionMode.CompositionMode_Lighten)
        colors = rgb_colors
        default_foreground = white
        default_background = black
        mode = "lighten"

    default_foreground = 0x888888
    root = w.get_root_frequency()
    all_svgs=[]  
    # we'll set the background with a svg rect
    svg = raw_svg_to_group('<rect width="1500" height="1500" />', default_background)
    all_svgs.append(svg)
    for check, notes in [(ui_opts.get_show_just_notes, just_intonation_notes),
                         (ui_opts.get_show_ET_notes, equal_temperament_notes)]:      
        if check():
            overlay = make_note_lines(
                root,
                {i: f * root for i, f in zip(twelve_tone_names, notes)},
                0.5, 600)
            svg = raw_svg_to_group(overlay, default_foreground)
            all_svgs.append(svg)
        
    if ui_opts.get_show_base_spiral():
        overlay = make_spiral_octave_lines(root)
        svg = raw_svg_to_group(overlay, default_foreground)
        all_svgs.append(svg)
        
    theta = 0
    width, height = ui_opts.virtual_size

    for notegroup, col in zip(w.notes, colors):
        notegrp_svg = make_spiral_lines_from_notes(root, notegroup)
        notegrp_svg += '<circle r="{}" cx="{}" cy="{}"/>'.format(
            width / 30.0, width / 10.0 + width / 45.0 * math.sin(theta),
            width / 10.0 + width / 45.0 * math.cos(theta))
        theta += math.pi*2.0/len(w.notes)
        # convert to a svg group with some extra tags to make inkscape happy
        svg = raw_svg_to_group(
            notegrp_svg, col, 
            extras='inkscape:groupmode="layer" filter="url(#blend)"')
        all_svgs.append(svg)
        
    # finally we'll render tham all
    for svg in all_svgs:
        render_svg(svg, qp)
    # try to save an inkscape compatible svg file.
    # we can add a darken/lighten filter, and we need to add
    # enable-background="new" to the svg header and the 
    # inkscape ns:
    with open("out.svg", 'w') as f: 
        f.write('<svg  xmlns="http://www.w3.org/2000/svg" '
                'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
                'version="1.1" width="{}" height="{}" '
                'enable-background="new">'.format(width, height))
        f.write('<filter id="blend">'
                '<feBlend in2="BackgroundImage" mode="{0}" />'
                '</filter>'.format(mode))
        f.write("\n".join(all_svgs))
        f.write("</svg>")

    d.widget().setPixmap(PySide.QtGui.QPixmap.fromImage(qim))
    # qim.save("out.png", 'PNG')
    qp = None  # we have to make sure qim is deleted before QPainter?
    
if __name__=="__main__":
    w=Waveform()
    play_waveform(w)
    if use_ui:
        ui_opts = UIOptions()
        f = Framework()
        f.get_main_window().resize(800,600)
        
        d=PySide.QtGui.QDockWidget("Note Visualization")
        d.setWidget(PySide.QtGui.QLabel())
        f.get_main_window().addDockWidget(PySide.QtCore.Qt.RightDockWidgetArea, d)
        
        # play notes is threaded, so we need to call notes_changed from the
        # ui thread.
        w.on_notes_changed.append(lambda: f.run_on_ui_thread(notes_changed))
        f.display_widgets([f.get_obj_widget(w), f.get_obj_widget(ui_opts)])
        f.close()
