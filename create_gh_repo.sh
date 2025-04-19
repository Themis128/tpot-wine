#!/bin/bash

REPO_NAME="tpot-wine"
REPO_DESC="Streamlit wine dashboard with Together AI bot"

# Ensure we're inside a git repo
if [ ! -d .git ]; then
  echo "Initializing new Git repo..."
  git init
fi

# Rename branch BEFORE first commit
git symbolic-ref HEAD refs/heads/main

# Stage & commit if nothing committed yet
if [ -z "$(git log --oneline 2>/dev/null)" ]; then
  echo "Creating initial commit..."
  git add .
  git commit -m "Initial commit: Streamlit wine app with Together AI bot"
fi

# Create remote GitHub repo using GitHub CLI
gh repo create "$REPO_NAME" \
  --private \
  --description "$REPO_DESC"

# Set remote and push to GitHub
git remote add origin "git@github.com:$(gh api user | jq -r .login)/$REPO_NAME.git"
git push -u origin main

echo "✅ Private GitHub repo '$REPO_NAME' initialized and pushed to 'main'."
