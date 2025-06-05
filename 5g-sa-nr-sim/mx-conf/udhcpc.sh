#!/bin/sh
#/*
# Copyright (C) 2021-2025 BubbleRAN SAS
#
# E2E 5G O-RAN RIC and xapp
# Last Changed: 2025-04-25
# Project: MX-ORS
# Full License: https://bubbleran.com/resources/files/BubbleRAN_Licence-Agreement-1.3.pdf)
#*/



case $1 in
    'bound'|'renew')
        busybox ifconfig "$interface" ${mtu:+mtu $mtu} "$ip" netmask "$subnet" ${broadcast:+broadcast $broadcast}
        crouter=$(busybox ip -4 route show dev "$interface" | busybox awk '$1 == "default" { print $3; }')
        router="${router%% *}"
        if [ ".$router" != ".$crouter" ]; then
            busybox ip -4 route flush exact 0.0.0.0/0 dev "$interface"
        fi
	;;

    'deconfig')
        busybox ip link set "$interface" up
        busybox ip -4 addr flush dev "$interface"
        busybox ip -4 route flush dev "$interface"
	;;

    'leasefail' | 'nak')
	;;

    *)
        echo "$0: Unknown udhcpc command: $1" >&2
        exit 1
	;;
esac

