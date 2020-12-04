#!/bin/bash
# todo: document gadget install

dtoverlay dwc2
modprobe dwc2
modprobe libcomposite
cd /sys/kernel/config/usb_gadget/
mkdir -p g1
cd g1
echo 0x1d6b > idVendor  # Linux Foundation
echo 0x0104 > idProduct # Multifunction Composite Gadget
echo 0x0100 > bcdDevice # v1.0.0
echo 0x0200 > bcdUSB    # USB2
mkdir -p strings/0x409
echo "HMPI200001" > strings/0x409/serialnumber
echo "HeadMouse Pi" > strings/0x409/manufacturer
echo "HeadMouse Pi" > strings/0x409/product

N="usb0"
# todo: rndis gadget for windows eth?
mkdir -p functions/ecm.$N
mkdir -p functions/hid.$N

echo "00:12:34:56:78:9a" > functions/ecm.$N/dev_addr
echo "00:a9:87:65:43:21" > functions/ecm.$N/host_addr

echo 1 > functions/hid.$N/protocol
echo 1 > functions/hid.$N/subclass
echo 8 > functions/hid.$N/report_length
echo -ne \\x05\\x01\\x09\\x06\\xa1\\x01\\x85\\x01\\x05\\x07\\x19\\xe0\\x29\\xe7\\x15\\x00\\x25\\x01\\x75\\x01\\x95\\x08\\x81\\x02\\x95\\x01\\x75\\x08\\x81\\x03\\x95\\x05\\x75\\x01\\x05\\x08\\x19\\x01\\x29\\x05\\x91\\x02\\x95\\x01\\x75\\x03\\x91\\x03\\x95\\x05\\x75\\x08\\x15\\x00\\x25\\x65\\x05\\x07\\x19\\x00\\x29\\x65\\x81\\x00\\xc0\\x05\\x01\\x09\\x02\\xa1\\x01\\x09\\x01\\xa1\\x00\\x85\\x02\\x05\\x09\\x19\\x01\\x29\\x03\\x15\\x00\\x25\\x01\\x95\\x03\\x75\\x01\\x81\\x02\\x95\\x01\\x75\\x05\\x81\\x03\\x05\\x01\\x09\\x30\\x09\\x31\\x16\\x01\\x80\\x26\\xff\\x7f\\x75\\x10\\x95\\x02\\x81\\x06\\xc0\\xc0\\x05\\x01\\x09\\x02\\xa1\\x01\\x09\\x01\\xa1\\x00\\x85\\x03\\x05\\x01\\x09\\x30\\x09\\x31\\x15\\x00\\x26\\xff\\x7f\\x75\\x10\\x95\\x02\\x81\\x02\\x09\\x38\\x15\\x81\\x25\\x7f\\x75\\x08\\x95\\x01\\x81\\x06\\xc0\\xc0   > functions/hid.$N/report_desc

C=1
mkdir -p configs/c.$C/strings/0x409
echo "Config $C: HeadMouse Pi" > configs/c.$C/strings/0x409/configuration
echo 500 > configs/c.$C/MaxPower
ln -s functions/ecm.$N configs/c.$C/
ln -s functions/hid.$N configs/c.$C/

ls /sys/class/udc > UDC

chown root:pi /dev/hidg0
chmod g+rw /dev/hidg0

ifup usb0
service dnsmasq restart
