#!/usr/bin/env bash
# CI's pip install, retried ONLY on network errors.
#
# PyPI read timeouts (pip's default is 15 s, 5 retries) have failed CI installs as a bogus
# "ResolutionImpossible": pip never saw the versions it timed out on. So: a longer timeout
# and more retries, and if pip still fails, retry the whole install, but only when its log
# shows a network error. A real dependency conflict has no network error in the log, and
# fails at once, as it should.
#
#   tools/ci-pip-install.sh -e '.[dev,geo]'
set -uo pipefail
export PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=10
backoff=${CI_PIP_BACKOFF_SECONDS:-30}   # seconds; tests set it to 0
network='ReadTimeoutError|ConnectTimeoutError|Read timed out|ConnectionResetError|RemoteDisconnected|IncompleteRead|NewConnectionError|ProtocolError'
# Without its log the script can't tell a network error from a real conflict: fatal.
log=$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/ci-pip-install.XXXXXX") || { echo "::error::cannot create a temporary file for pip's log"; exit 2; }
for attempt in 1 2 3; do
    python -m pip install -v "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && exit 0
    if ! grep -qE "$network" "$log"; then
        echo "::error::pip install failed with no network error in its log: a real dependency problem, not retried"
        exit "$rc"
    fi
    echo "::warning::pip install failed after network errors (attempt $attempt of 3)"
    [ "$attempt" -lt 3 ] && sleep $((backoff * attempt))
done
exit "$rc"
