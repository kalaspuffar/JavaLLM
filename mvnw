#!/bin/sh
# Maven Wrapper script — downloads Maven if not already cached, then runs it.
# Based on the official Maven Wrapper (https://maven.apache.org/wrapper/).

set -e

# Determine project base directory
if [ -z "$MAVEN_PROJECTBASEDIR" ]; then
  MAVEN_PROJECTBASEDIR=$(cd "$(dirname "$0")" && pwd)
  export MAVEN_PROJECTBASEDIR
fi

WRAPPER_PROPERTIES="$MAVEN_PROJECTBASEDIR/.mvn/wrapper/maven-wrapper.properties"
if [ ! -f "$WRAPPER_PROPERTIES" ]; then
  echo "Error: Could not find $WRAPPER_PROPERTIES" >&2
  exit 1
fi

# Read distributionUrl from properties
DISTRIBUTION_URL=$(grep 'distributionUrl' "$WRAPPER_PROPERTIES" | cut -d'=' -f2-)

# Maven home in user's cache
MAVEN_VERSION=$(echo "$DISTRIBUTION_URL" | sed 's|.*/apache-maven-\(.*\)-bin\.zip|\1|')
MAVEN_HOME="$HOME/.m2/wrapper/dists/apache-maven-$MAVEN_VERSION"

if [ ! -d "$MAVEN_HOME/bin" ]; then
  echo "Downloading Maven $MAVEN_VERSION..."
  DOWNLOAD_DIR=$(mktemp -d)
  ZIP_FILE="$DOWNLOAD_DIR/maven.zip"

  if command -v curl > /dev/null 2>&1; then
    curl -fsSL -o "$ZIP_FILE" "$DISTRIBUTION_URL"
  elif command -v wget > /dev/null 2>&1; then
    wget -q -O "$ZIP_FILE" "$DISTRIBUTION_URL"
  else
    echo "Error: Neither curl nor wget found. Cannot download Maven." >&2
    exit 1
  fi

  mkdir -p "$MAVEN_HOME"
  unzip -q -o "$ZIP_FILE" -d "$DOWNLOAD_DIR"
  mv "$DOWNLOAD_DIR"/apache-maven-*/* "$MAVEN_HOME/"
  rm -rf "$DOWNLOAD_DIR"
  echo "Maven $MAVEN_VERSION installed to $MAVEN_HOME"
fi

export M2_HOME="$MAVEN_HOME"
export PATH="$MAVEN_HOME/bin:$PATH"

exec mvn "$@"
