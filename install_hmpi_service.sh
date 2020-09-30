#!/bin/bash

HMPIDIR=/hmpi

grep "^dtoverlay=dwc2" /boot/config.txt ||
    echo -e "\n# Installed by headmouse pi installer\ndtoverlay=dwc2" | sudo tee -a /boot/config.txt
grep "^dwc2" /etc/modules || echo "dwc2" | sudo tee -a /etc/modules
grep "^libcomposite" /etc/modules || echo "libcomposite" | sudo tee -a /etc/modules

sudo chmod +x ${HMPIDIR}/headmousepi_gadgets.sh
sudo chmod +x ${HMPIDIR}/uninstall_hmpi_service.sh

if ! systemctl | grep headmousepi-gadgets; then
  sudo cp ${HMPIDIR}/headmousepi-gadgets.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl start headmousepi-gadgets
  sudo systemctl enable headmousepi-gadgets
fi

if ! systemctl | grep headmouse-pi; then
  sudo cp ${HMPIDIR}/headmouse-pi.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl start headmouse-pi
  sudo systemctl enable headmouse-pi
fi
