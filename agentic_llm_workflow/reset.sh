#!/bin/bash

# Determine whether to use sudo-- docker commands are executed without sudo
if [ -z "$BUBBLERAN_HOST_PWD" ]; then
  SUDO='sudo'
else
  SUDO=''
fi

#
cd 5g-sa-nr-sim/
docker compose down
$SUDO rm -rf sqlite3/data

cd ../5g-sa-usrp/
docker compose down
$SUDO rm -rf sqlite3/data

$SUDO rm ../data/persistent_db
$SUDO rm ../data/historical_db

echo "Cleanup complete"