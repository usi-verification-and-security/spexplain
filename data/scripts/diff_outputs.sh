#!/bin/bash

## git-diff experimental outputs ignoring times and missing contents
## > we can check the results even if the experiments have not finished yet

cd "$(dirname $0)/../explanations"
git diff | grep '^+' | grep -Ev '(^\+(real|user|sys)|\.time\.txt$)' | less
