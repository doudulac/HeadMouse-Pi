---
layout: page
permalink: /downloads/
---
# Downloads

Headmouse Pi is provided as a pre-built image that can be installed onto a microSD card. The image
contains everything necessary to make a [Raspberry Pi 4B](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)
function as a HeadMouse Pi. It is primarily a [Raspberry Pi OS Lite](https://www.raspberrypi.org/software/operating-systems/)
image with `headmouse.py` and the supporting software added. The image is generated using
[pi-gen](https://github.com/RPi-Distro/pi-gen#pi-gen), the same "tool used to create Raspberry Pi OS
images."

## Current image (**{{ site.github.latest_release.tag_name }}**)
{% assign asset = site.github.latest_release.assets | where: "content_type", "application/zip" | first%}

| **Image File** | [<i class="fas fa-download"></i> {{ asset.name }}]({{ asset.browser_download_url }}) |
| **Size** | {{ asset.size | divided_by: 1000000 }}MB |
| **Created** | {{ asset.created_at | date: "%s" | date: "%D %r %Z" }} |
| **Details** | {{ site.github.latest_release.body }} |
