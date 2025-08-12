#!/usr/bin/env sh
COMMIT=$1
QUALITY=$2
WIN_CODE_CMD=$3
APP_NAME=$4
DATAFOLDER=$5

shift 5

if [ "$VSCODE_WSL_DEBUG_INFO" = true ]; then
	set -x
fi

# Read stdin
if [ ! -t 0 ]; then
	for var in "$@"
	do
		if [ "$var" = "-" ]; then
			PIPE_STDIN_FILE=$(mktemp /tmp/cursor-stdin-XXX)
			while IFS= read -r line; do
				printf "%s\n" "$line" >> "$PIPE_STDIN_FILE"
			done
		fi
	done
fi

VSCODE_REMOTE_BIN="$HOME/$DATAFOLDER/bin"
AUTHORITY="wsl+default"

if [ "$WSL_DISTRO_NAME" ]; then
	AUTHORITY="wsl+$WSL_DISTRO_NAME"
else
	echo "Please update your version of WSL by updating Windows 10 to the May 19 Update, version 1903, or later.";
  exit 1
fi

"$(dirname "$0")/wslDownload.sh" "$COMMIT" "$QUALITY" "$VSCODE_REMOTE_BIN"
RC=$?;
if [ $RC -ne 0 ]; then
	exit $RC
fi

STORED_ENV=$(mktemp /tmp/vscode-distro-env.XXXXXX)
env --null > "$STORED_ENV"

VSCODE_CLIENT_COMMAND="$WIN_CODE_CMD" \
VSCODE_CLIENT_COMMAND_CWD="$(dirname "$0")" \
VSCODE_CLI_AUTHORITY="$AUTHORITY" \
VSCODE_CLI_REMOTE_ENV="$STORED_ENV" \
VSCODE_STDIN_FILE_PATH="$PIPE_STDIN_FILE" \
WSLENV="VSCODE_CLI_REMOTE_ENV/w:$WSLENV" \
"$VSCODE_REMOTE_BIN/$COMMIT/bin/remote-cli/$APP_NAME" "$@"
