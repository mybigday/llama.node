#!/bin/bash

set -e

# General

yarn clean && yarn build-native -C
