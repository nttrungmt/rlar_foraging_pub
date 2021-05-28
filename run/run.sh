# Usage: bash run.sh <port> <num_robots>

#source /opt/ros/kinetic/setup.bash
source devel/setup.bash
export ARGOS_PLUGIN_PATH=$ARGOS_PLUGIN_PATH:./devel/lib

HOST=$1
PORT_NUM=$2
NUM_ROBOTS=$3
if [[ -z "$4" ]]; then
    ARENA_LEN=9.0
    FOOD_POSY=2.7
    WALL_H_POSY=4.0
    WALL_V_LEN=8.0
else
    ARENA_LEN=$4
    FOOD_POSY=$(python -c "print($4 * 0.3)")
    WALL_V_LEN=$(python -c "print($4 - 1)")
    WALL_H_POSY=$(python -c "print(${WALL_V_LEN} * 0.5)")
fi
#FOOD_POSY=expr $ARENA_LEN \* 0.3

TEMPLATE='./src/ma_foraging/argos_worlds/template.argos'
OUTFILE="./src/ma_foraging/argos_worlds/test_$2_$3_${ARENA_LEN}.argos"

sed -e "s|HOST_ADDRESS|${HOST}|g" -e "s|PORT_NUM|${PORT_NUM}|g" -e "s|NUM_ROBOTS|${NUM_ROBOTS}|g" -e "s|ARENA_LEN|${ARENA_LEN}|g" -e "s|FOOD_POSY|${FOOD_POSY}|g" $TEMPLATE -e "s|WALL_V_LEN|${WALL_V_LEN}|g" -e "s|WALL_H_POSY|${WALL_H_POSY}|g" > $OUTFILE

argos3 -c ${OUTFILE}
