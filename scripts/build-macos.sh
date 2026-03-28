#!/bin/bash

set -e

# Ensure a consistent macOS deployment target.
#
# This must live in the macOS build script (not the GitHub Actions workflow)
# so local builds and CI behave the same.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
export CMAKE_OSX_DEPLOYMENT_TARGET="${CMAKE_OSX_DEPLOYMENT_TARGET:-${MACOSX_DEPLOYMENT_TARGET}}"

# General

npx cmake-js rebuild -C --CDTO_PACKAGE=ON
