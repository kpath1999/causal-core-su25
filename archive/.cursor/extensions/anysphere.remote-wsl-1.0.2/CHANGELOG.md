# Cursor Remote WSL Changelog

## v1.0.2
- Add "Open Folder in WSL" command

## v1.0.1
- Fix an issue where launching Cursor from with WSL shell opened in a non-remote window. Now, running the `code` or `cursor` from within a WSL directory (in a WSL shell) will open as a remote window.

## v1.0.0
- Bug fixes and improvements

## v0.0.11

- Added prompt to reinstall the server on failed connections
- Removed the clean install when a stuck connection requires a server reboot and window reload. Now, just the running servers are killed. The server binaries are left in place.
- Added Kill Server and Reload Window Command
- Added Reinstall Server and Reload Window Command
- Added cleanup of old server binaries


## v0.0.10

- Added telemetry (enabled when privacy mode is disabled)


## v0.0.9

- Updated wording in error message
