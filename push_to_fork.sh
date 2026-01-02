#!/bin/bash
# Script to push code to your GitHub fork
# Usage: ./push_to_fork.sh YOUR_GITHUB_USERNAME

if [ -z "$1" ]; then
    echo "Usage: ./push_to_fork.sh YOUR_GITHUB_USERNAME"
    echo "Example: ./push_to_fork.sh myusername"
    exit 1
fi

GITHUB_USER=$1
FORK_URL="https://github.com/${GITHUB_USER}/profile-photo.git"

echo "Setting remote to your fork: ${FORK_URL}"
git remote set-url origin "${FORK_URL}"

echo "Verifying input/output folders are ignored..."
if git check-ignore profile_photo/input/ profile_photo/output/ > /dev/null 2>&1; then
    echo "✓ Input and output folders are properly ignored"
else
    echo "✗ WARNING: Input/output folders may not be ignored!"
    exit 1
fi

echo "Pushing to your fork..."
git push -u origin main

echo "Done! Your code is now on your fork at: ${FORK_URL}"

