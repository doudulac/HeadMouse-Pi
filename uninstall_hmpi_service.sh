#!/bin/bash

SYSDIR=/sys/kernel/config/usb_gadget
DEVDIR=$SYSDIR/g1
HMPIG=/etc/systemd/system/headmousepi-gadgets.service
HMPI=/etc/systemd/system/headmouse-pi.service

sudo systemctl stop headmouse-pi
sudo systemctl stop headmousepi-gadgets

echo '' | sudo tee $DEVDIR/UDC

echo "Removing strings from configurations"
for dir in "$DEVDIR"/configs/*/strings/*; do
	[ -d "$dir" ] && sudo rmdir "$dir"
done

echo "Removing functions from configurations"
for func in "$DEVDIR"/configs/*.*/*.*; do
	[ -e "$func" ] && sudo rm "$func"
done

echo "Removing configurations"
for conf in "$DEVDIR"/configs/*; do
	[ -d "$conf" ] && sudo rmdir "$conf"
done

echo "Removing functions"
for func in "$DEVDIR"/functions/*.*; do
	[ -d "$func" ] && sudo rmdir "$func"
done

echo "Removing strings"
for str in "$DEVDIR"/strings/*; do
	[ -d "$str" ] && sudo rmdir "$str"
done

echo "Removing gadget"
[ -d "$DEVDIR" ] && sudo rmdir "$DEVDIR"

echo "Removing headmousepi-gadgets.service"
[ -e "$HMPIG" ] && sudo systemctl disable headmousepi-gadgets && sudo rm "$HMPIG"

echo "Removing headmouse-pi.service"
[ -e "$HMPI" ] && sudo systemctl disable headmouse-pi && sudo rm "$HMPI"
