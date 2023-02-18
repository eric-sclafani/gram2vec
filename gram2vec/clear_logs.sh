#!/usr/bin/env bash

if find ../gram2vec/logs/ -mindepth 1 -maxdepth 1 | read; then
    echo "g2v: Old logs detected. Clearing..."
    rm ../gram2vec/logs/*
    echo "g2v: Done"
fi