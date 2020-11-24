---
layout: page
permalink: /downloads/
---
# Downloads

Headmouse Pi is provided as a pre-built binary image file that can be easily installed onto a
microSD card using the [Raspberry Pi Imager](https://www.raspberrypi.org/software/).
The image is compressed into a _*.zip_ file and contains everything necessary to make a
[Raspberry Pi 4B](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)
function as a HeadMouse Pi. The contents of the image are: first, a standard build of the 
[Raspberry Pi OS Lite](https://www.raspberrypi.org/software/operating-systems/) operating system,
and second, the HeadMouse Pi and additional supporting software. The image is generated using
[pi-gen](https://github.com/RPi-Distro/pi-gen#pi-gen), the same "tool used to create Raspberry Pi OS
images."

## Current image (**{{ site.github.latest_release.tag_name }}**)
{% assign asset = site.github.latest_release.assets | where: "content_type", "application/zip" | first%}

| **Image File** | [<i class="fas fa-download"></i> {{ asset.name }}]({{ asset.browser_download_url }}) |
| **Size** | {{ asset.size | divided_by: 1000000 }}MB |
| **Created** | {{ asset.created_at | date: "%s" | date: "%D %r %Z" }} |
| **Details** | {{ site.github.latest_release.body }} |
