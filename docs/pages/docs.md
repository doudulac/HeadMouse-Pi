---
layout: page
permalink: /docs/
---
# Documentation
* toc
{:toc}


# Getting Started
The Headmouse Pi requires some hardware assembly and a software install.  Being built on top of a
Raspberry Pi 4B means that several pieces need to be purchased individually and assembled.  These
components are readily available on Amazon or other online e-tailers.  The software is all open 
source and freely available online.

# Build Instructions
Raspberry Pis come in various configurations of RAM and can be assembled inside numerous different 
cases.  The following hardware requirements will detail one set of hardware that has been tested, 
although other combinations may work as long as the Raspberry Pi is a model 4B (model 1/2/3 won't
work) and has at leaat 2GB ram.

## Hardware Requirements
* Raspberry Pi 4B 2GB ram or better (see https://www.raspberrypi.org/documentation/ for more info)
* 4GB or better microSD card
* Argon NEO Pi 4 Raspberry Pi Case
* USB Webcam or Raspberry Pi Camera Module V2
* SD Card reader/writer for programming microSD card (check on laptop, computer for built-in or
  use USB/microSD card reader adapter)
* USB-C cable (Like most any phone/tablet USB-C charge cable)
* Power Enhancer Y (1 Female to 2 Male) Data Charge Cable Extension Cord [1]
* USB/AC wall charger (5V, 2-3 Amps) Newer phone and tablet USB chargers should be 2Amps. [1]

[1] may not be needed with some hosts, e.g. MacBook Pro


### Where to Buy Hardware
Adafruit is an American company that supports the DIY and maker communities and carries a lot of
Raspberry Pi components.

  * Adafruit: [Raspberry Pi 4 Model B - 2 GB RAM](https://www.adafruit.com/product/4292)
  * Adafruit: [Raspberry Pi Camera Board v2 - 8 Megapixels](https://www.adafruit.com/product/3099)
  * Adafruit: [Raspberry Pi NoIR Camera Board v2 - 8 Megapixels](https://www.adafruit.com/product/3100)

  * Amazon: [Argon NEO Raspberry Pi 4 Model B Heatsink Case](https://www.amazon.com/Argon-Raspberry-Heatsink-Supports-Accessible/dp/B07WMG27T7/ref=sr_1_1)
  * Amazon: [SanDisk 32GB Extreme microSDHC UHS-I Memory Card with Adapter](https://www.amazon.com/SanDisk-Extreme-microSDHC-UHS-3-SDSQXAF-032G-GN6MA/dp/B06XWMQ81P/ref=sr_1_1)
  
  * Amazon: [MMNNE USB 3.0 Female to Dual USB Male Extra Power Data Y Extension Cable](https://www.amazon.com/MMNNE-Female-Extra-Extension-Mobile/dp/B06XPL75R9/ref=sr_1_1)
  * Amazon: [HIGHROCK 30cm USB 2.0 a Power Enhancer Y 1 Female to 2 Male Data Charge Cable](https://www.amazon.com/HIGHROCK-Enhancer-Female-Charge-Extension/dp/B00NIGO4NM/ref=sr_1_1)


### Assembling the Hardware 
* Coming soon!

## Software Requirements 
The following lists the key software components that comprise HeadMouse Pi. These are all included 
in the downloadable pre-built image, so there’s no need to acquire these separately unless
performing a manual install.

* Linux Operating System
  * configfs (gadget mode)
* Python 3
  * dlib eventlet filterpy flask flask-socketio flask-wtf imutils numpy opencv-python picamera
    scipy systemd-python yappi
* Supporting library dependencies
  * libatlas3-base libavcodec58 libavformat58 libgtk-3-0 libilmbase23 libopenexr23 libopenjp2-7 
    libsm6 libswscale5 libtiff5 libwebp6
* dnsmasq
* headmouse.py


### Installing the Software
The easiest way to install Headmouse Pi is to use the Raspberry Pi Imager to flash the pre-built
image onto an SD card. Continue to [[Installing Pre-Built Image](#installing-pre-built-image)].


### Installing Pre-Built Image
* Download and Install [Raspberry Pi Imager](https://www.raspberrypi.org/software/){:target="_blank"}
  for your machine (Windows, MacOS, Linux)
* Download [Headmouse Pi Image]({{ '/downloads/' | relative_url }}){:target="_blank"}
* Insert microSD card into SD card reader or use a USB-microSD card adapter

<iframe width="560" height="315" src="https://www.youtube.com/embed/dnUA0y-LO54" frameborder="0" 
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
allowfullscreen></iframe>

* Launch Raspberry Pi Imager
* Click “Choose OS”
* Scroll down and select "Use custom"
* Locate and choose the downloaded Headmouse Pi zip file
* Click “Choose SD Card”
* Locate and choose the inserted microSD card
* Click “WRITE” (Then YES to continue). This will take several minutes. Click CONTINUE when 
  prompted. The microSD card is now ready to go!


### Installing on Existing Raspberry Pi OS
* Coming soon!


# User Interface
* Coming soon!


## Connecting to HeadMouse Pi
* Coming soon!


## User Preferences 
* Coming soon!


## Camera View
* Coming soon!


## Debug Data
* Coming soon!


# Prototype Hardware 
* Raspberry Pi 4B 2GB ram ($35)
* 4GB or larger microSD card ($6)
* Raspberry Pi Camera Module V2 (Noir version $25)
* Argon NEO Pi 4 Raspberry Pi Case ($15)


# Usability
Use of the HeadMouse Pi requires motor control of head and neck (side to side and up/down) and use 
of facial gestures (raising eyebrows, opening mouth).  

| **Function** | **Facial Gesture** |
| Left-click   | Raise eyebrows |
| Double-click | Raise eyebrows 2X |
| Right-click  | Open mouth wide |
| Drag-and-drop| Raise eyebrows > 1sec (“sticky click”)* |
| Pause/unpause mouse | Close eyes > 3sec while facing camera |

*enabled by default (can be disabled upon launch)


# Future Plans

* Provide detailed design and installation instructions (with pictures)
* Provide user’s guide with troubleshooting tips
* Document process of adding external buttons to activate mouse-clicks
* Add additional gestures to include mouse wheel and swiping actions
* Allow user-configurable facial gestures to click actions
* Develop user interface for easy configuration
* Testing and improvement on Windows platform (original development on a Mac)
* Testing and integration with popular gaming consoles (e.g., Nintendo, Playstation)
* Integration of Wiimote for control
* Add RNDIS ethernet gadget for Windows
* Test on Raspberry Pi Zero
* Develop pupil tracking for eye-gaze control
