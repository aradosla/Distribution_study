# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get if we're located in a docker and miniforge exists
FILE=/usr/local/DA_study/miniforge_docker/bin/activate

# Check if  the job is launched from docker
if [[ -f "$FILE" ]];

# Source the make_miniforge.sh script
then
    source $FILE
# Activate miniforge from afs
else
    #source $SCRIPT_DIR/miniforge/bin/activate
    source /afs/cern.ch/work/a/aradosla/private/miniforge3/bin/activate

fi


