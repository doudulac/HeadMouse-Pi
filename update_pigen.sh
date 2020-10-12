#!/usr/bin/env bash

DIR=$(dirname "${BASH_SOURCE[0]}")
PIGEN=~/src/pi-gen
FILES="$DIR/headmouse* $DIR/shape* $DIR/requirements.txt "
FILES+="$DIR/dnsmasq* $DIR/interface* "
rm "$PIGEN"/headmousepi/00-install-hmpi/files/*
cp -v $FILES "$PIGEN"/headmousepi/00-install-hmpi/files/
