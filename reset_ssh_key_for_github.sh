#!/bin/bash

KEY_PATH="$HOME/.ssh/id_ed25519"

echo "== SSH Key Reset for GitHub =="

# Step 1: Backup old key if it exists
if [ -f "$KEY_PATH" ]; then
    echo "Backing up old SSH key..."
    mv "$KEY_PATH" "${KEY_PATH}_backup"
    mv "${KEY_PATH}.pub" "${KEY_PATH}_backup.pub"
fi

# Step 2: Generate new key
echo "Generating a new SSH key..."
read -p "Enter your GitHub email: " github_email
ssh-keygen -t ed25519 -C "$github_email" -f "$KEY_PATH"

# Step 3: Add key to SSH agent
echo "Adding new SSH key to agent..."
eval "$(ssh-agent -s)"
ssh-add "$KEY_PATH"

# Step 4: Print public key to console
echo
echo "=== COPY THIS KEY TO GITHUB ==="
echo
cat "${KEY_PATH}.pub"
echo
echo "Visit: https://github.com/settings/keys to paste your key."
echo

# Step 5: Test connection
read -p "Press Enter to test SSH connection to GitHub..."
ssh -T git@github.com

echo
echo "✅ Done. Try pushing with: git push --set-upstream origin main"
