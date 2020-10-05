#!/usr/bin/env bash

DIR=$(dirname "${BASH_SOURCE[0]}")

if [ x"$1" = "x" ]; then
  echo "Usage: $(basename "$0") <dest pi name or ip address>"
  exit 1
fi

FILES="$DIR/head* $DIR/inst* $DIR/unins*"

ssh "$1" "[ ! -d /hmpi ] && sudo mkdir /hmpi && sudo chown pi:pi /hmpi"

ssh "$1" test -e /hmpi/shape_predictor_68_face_landmarks.dat || FILES+=" $DIR/shape*"

scp $FILES pi@"$1":/hmpi/
