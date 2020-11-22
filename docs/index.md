---
layout: default
---
# HeadMouse Pi

The HeadMouse Pi is a low cost, universal, head-tracking mouse built from a Raspberry Pi and
Camera Module or ordinary webcam. Its goal is to be a viable computer access technology for me
and others who lack the ability to use their hands. It offers a low-cost and no-insurance-required 
alternative to larger and more sophisticated devices.  It requires no special apparatus to be worn 
(e.g. glasses or reflective dots) by the user; instead, it tracks your head in space and translates 
the movement into a USB mouse protocol.
 

# Prototype Hardware

*   Raspberry Pi 4B 2GB ram ($35)
*   2GB or larger microSD card ($6)
*   Raspberry Pi Camera Module V2 (Noir version $25)
*   Argon NEO Pi 4 Raspberry Pi Case ($15)


# Usability

Use of the HeadMouse Pi requires motor control of head and neck (side to side and up/down) and use 
of facial gestures (raising eyebrows, opening mouth).  

<table>
  <tr>
   <td><strong>Function</strong>
   </td>
   <td><strong>Facial Gesture</strong>
   </td>
  </tr>
  <tr>
   <td>Left-click
   </td>
   <td>Raise eyebrows
   </td>
  </tr>
  <tr>
   <td>Double-click
   </td>
   <td>Raise eyebrows 2X
   </td>
  </tr>
  <tr>
   <td>Right-click
   </td>
   <td>Open mouth wide
   </td>
  </tr>
  <tr>
   <td>Drag-and-drop
   </td>
   <td>Raise eyebrows > 1sec (“sticky click”)*
   </td>
  </tr>
  <tr>
   <td>Pause/unpause mouse
   </td>
   <td>Close eyes > 3sec while facing camera
   </td>
  </tr>
</table>
*enabled by default (can be disabled upon launch)


# Future Plans

*   Provide detailed design and installation instructions (with pictures)
*   Provide user’s guide with troubleshooting tips
*   Document process of adding external buttons to activate mouse-clicks
*   Add additional gestures to include mouse wheel and swiping actions
*   Allow user-configurable facial gestures to click actions
*   Develop user interface for easy configuration
*   Testing and improvement on Windows platform (original development on a Mac)
*   Testing and integration with popular gaming consoles (e.g., Nintendo, Playstation)
*   Integration of Wiimote for control
*   Add RNDIS ethernet gadget for Windows
*   Test on Raspberry Pi Zero
*   Develop pupil tracking for eye-gaze control
