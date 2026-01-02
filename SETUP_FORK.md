# Setting up GitHub Fork

## Option 1: Create Fork via GitHub Web Interface (Recommended)

1. Go to https://github.com/rnag/profile-photo
2. Click the "Fork" button in the top right
3. Choose your account/organization for the fork
4. Once forked, note your fork's URL (e.g., `https://github.com/YOUR_USERNAME/profile-photo`)

Then run these commands:

```bash
# Update the remote to point to your fork
git remote set-url origin https://github.com/YOUR_USERNAME/profile-photo.git

# Push your changes
git push -u origin main
```

## Option 2: Create Fork via GitHub CLI

1. First authenticate:
```bash
gh auth login
```

2. Then create the fork:
```bash
gh repo fork rnag/profile-photo --clone=false
```

3. Update remote and push:
```bash
# Get your GitHub username (or set it manually)
GITHUB_USER=$(gh api user --jq .login)
git remote set-url origin https://github.com/${GITHUB_USER}/profile-photo.git
git push -u origin main
```

## Verify Input/Output Folders are Excluded

The following folders are in `.gitignore` and will NOT be uploaded:
- `profile_photo/input/` - Your personal images
- `profile_photo/output/` - Processed output images

You can verify with:
```bash
git check-ignore profile_photo/input/ profile_photo/output/
```

Both should return the folder paths, confirming they're ignored.

