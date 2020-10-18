#!/usr/bin/env bash

DIR=$(dirname "${BASH_SOURCE[0]}")
PIGEN=~/src/pi-gen/headmousepi/00-install-hmpi/files

if [ x"$1" = "x" ]; then
  echo "Usage: $(basename "$0") pigen|<dest pi name or ip address>"
  exit 1
elif [ "$1" = "pigen" ]; then
    DEST=$PIGEN/
else
    DEST=pi@$1:/hmpi
    SSH="-e \"ssh -l pi\""
    ssh "$1" "[ ! -d /hmpi ] && sudo mkdir /hmpi && sudo chown pi:pi /hmpi"
fi

EXCLUDES="--exclude \".*\" "
EXCLUDES+="--exclude __pycache__/ "
EXCLUDES+="--exclude hm.log "
EXCLUDES+="--exclude deploy.sh "

eval rsync -av --delete-after $EXCLUDES $SSH $DIR/ $DEST
