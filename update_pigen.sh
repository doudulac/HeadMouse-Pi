#!/usr/bin/env bash

DIR=$(dirname "${BASH_SOURCE[0]}")
PIGEN=~/src/pi-gen
FILES="$DIR/head* $DIR/inst* $DIR/unins* $DIR/shape* $DIR/requirements.txt"
cp -v $FILES "$PIGEN"/headmousepi/00-install-hmpi/files/
