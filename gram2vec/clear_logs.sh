#!/usr/bin/env bash

if find ../gram2vec/logs/ -mindepth 1 -maxdepth 1 | read; then
    rm ../gram2vec/logs/*
fi