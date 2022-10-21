#!/bin/bash

sopath=/lib/x86_64-linux-gnu

if [ ! -L ${sopath}/libnvcuvid.so ]; then
    echo Create soft link ${sopath}/libnvcuvid.so.1
    ln -s ${sopath}/libnvcuvid.so.1 ${sopath}/libnvcuvid.so
fi

if [ ! -L ${sopath}/libnvidia-encode.so ]; then
    echo Create soft link ${sopath}/libnvidia-encode.so.1
    ln -s ${sopath}/libnvidia-encode.so.1 ${sopath}/libnvidia-encode.so
fi