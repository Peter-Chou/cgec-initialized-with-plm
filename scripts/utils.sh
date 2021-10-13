#!/bin/bash

PRESELECT_RATIO=0.15

function usage()
{
    echo "if this was a real script you would see something useful here"
    echo ""
    echo "\t--help"
    echo "\t--gpu=$GPU"
    echo ""
}

function train_args() {
    GPU=0
    MODE=""
    MODEL_NAME=""
    TRAIN_RATIO=1

    for ARGUMENT in "$@"
    do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	VALUE=$(echo $ARGUMENT | cut -f2 -d=)

	case "$KEY" in
            --help)
		usage
		exit
		;;
	    --gpu)
		GPU=${VALUE}
		;;
	    --model_name)
		MODEL_NAME=${VALUE}
		;;
	    --mode) # formal / preselect
		if [[ "$VALUE" =~ ^(formal|preselect)$ ]]; then
		    MODE=${VALUE}
		    if [[ "$VALUE" == "preselect" ]]; then
			TRAIN_RATIO=${PRESELECT_RATIO}
		    fi
		else
		    echo "mode must be formal or preselect"
		    exit 1
		fi
		;;
	    *)
		echo "ERROR: unknown parameter \"$PARAM\""
		usage
		exit 1
		;;
	esac
    done

    if [ -z $MODE ]
    then
	echo "--mode must be given."
	exit 1
    fi

    if [ -z $MODEL_NAME ] #ckpt is empty
    then
	echo "--model_name must be given."
	exit 1
    fi
}

function test_args() {
    GPU=0
    CKPT=""
    TEST_MODE=""
    TEST_MODEL_NAME=""

    for ARGUMENT in "$@"
    do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	VALUE=$(echo $ARGUMENT | cut -f2 -d=)

	case "$KEY" in
            --help)
		usage
		exit
		;;
	    --gpu)
		GPU=${VALUE}
		;;
	    --mode) # formal / preselect
		if [[ "$VALUE" =~ ^(formal|preselect)$ ]]; then
		    TEST_MODE=${VALUE}
		else
		    echo "mode must be formal or preselect"
		    exit 1
		fi
		;;
	    --model_name)
		TEST_MODEL_NAME=${VALUE}
		;;
	    --ckpt)
		CKPT=${VALUE}
		;;
            *)
		echo "ERROR: unknown parameter \"$PARAM\""
		usage
		exit 1
		;;
	esac
    done

    if [ -z $TEST_MODE ]
    then
	echo "--mode must be given."
	exit 1
    fi

    if [ -z $TEST_MODEL_NAME ] #ckpt is empty
    then
	echo "--model_name must be given."
	exit 1
    fi

    if [ -z $CKPT ] #ckpt is empty
    then
	echo "--ckpt must be given."
	exit 1
    fi
}
