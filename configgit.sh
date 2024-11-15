#!/bin/bash

# Hardcoded Git user email and name
git_email="wooseokkoo@hotmail.com"
git_name="wesleykoo"

# Set the global Git configuration
git config --global user.email "$git_email"
git config --global user.name "$git_name"

# Verify the configuration
echo "Git configuration updated with the following values:"
echo "Email: $git_email"
echo "Name: $git_name"
git config --global --list