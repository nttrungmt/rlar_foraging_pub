source /opt/ros/kinetic/setup.bash
source devel/setup.bash
export ARGOS_PLUGIN_PATH=$ARGOS_PLUGIN_PATH:./devel/lib

#killall -9 argos3
#killall -9 roscore
#killall -9 rosmaster
#killall -9 rosout

#roscore &
#catkin_make && 
argos3 -c src/ma_foraging/argos_worlds/foraging.argos &

python2 src/ma_foraging/scripts/ForagingExp.py --agent=Hoff --num-episodes=1500 --steps-per-episode=150 
#--restore --last-timesteps=500 --restore-log-path=src/ma_foraging/log/tf_board/Hoff/20200221_205000_100_150
#--dumpExps

#python2 src/ma_foraging/scripts/ForagingExp.py --agent=DQN --num-episodes=500 --steps-per-episode=150 --batch-size=1024 --hidden-size=512 --lr=0.000001 --epsilon-decay=0.99 --epsilon-start=0.99 --buffer-size=500000 
#--restore --last-timesteps=300 --restore-log-path=src/ma_foraging/log/tf_board/DQN/20200219_213513_500_150 --restore-model-path=src/ma_foraging/log/DQN/ma-foraging-dqn_500_150_1024_32.mdl

############################################# windows script ############################################################
python .\scripts\ForagingExp.py --agent=Hoff --port=12345 --num-episodes=3000 --steps-per-episode=150 --base-path=.
#--dumpExps

#train
#python scripts\ForagingExp.py --agent=DQN --port=10000 --num-episodes=1500 --steps-per-episode=150 --batch-size=32 --num-batch-update-steps=16 --hidden-size=256 --lr=0.0001 --epsilon-decay=0.995 --epsilon-start=1.0 --buffer-size=100000 --base-path=. --use-per --lr-decay --replay-interval=5 --norm-data --use-neighbor-image --model-prefix-name=Pen0.1GpuObsv2Norm

#python scripts\ForagingExp.py --agent=DQN --port=12345 --num-episodes=3000 --steps-per-episode=150 --batch-size=32 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-decay=0.9975 --epsilon-start=1.0 --buffer-size=100000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --model-prefix-name=Pen0.1GpuObsv2Norm

#python scripts\ForagingExp.py --agent=DQN --port=23456 --num-episodes=3000 --steps-per-episode=150 --batch-size=32 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-decay=0.9975 --epsilon-start=1.0 --buffer-size=100000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --model-prefix-name=Pen0.1zGpuObsv2Norm

python scripts\ForagingExp.py --agent=DQN --port=12345 --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image 
--hidden-features-mode=1    #for no HF used
--hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5   #for no HF used
#--local-team-reward-rate=0.5

python scripts\ForagingExp.py --agent=DQN --port=23456 --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --model-prefix-name=Pen0.1Obsv2z0_3291_6452_6432 --cnn-layers=[[32,9,1],[64,5,2],[64,3,2]]

python scripts\ForagingExp.py --agent=DQN --port=34567 --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --model-prefix-name=Pen0.1Obsv2z1_3291_6452_6432 --cnn-layers=[[32,9,1],[64,5,2],[64,3,2]]

#TRAIN FullHF port 10000
python scripts\ForagingExp.py --agent=DQN --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image  --hidden-features-mode=0    --port=10000 

#TRAIN NoHF port 12345
python scripts\ForagingExp.py --agent=DQN --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=1    --port=12345 

#TRAIN RLaR_HF_2 port 23456
python scripts\ForagingExp.py --agent=DQN --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --port=23456 

#TRAIN RLaR_HF_1 port 34567
python scripts\ForagingExp.py --agent=DQN --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=1 --mdn-model-path=.\save\mdn2L_32x5x2_32x5x2_32x3x2_32_1_256_500_mse_0.0001_0.01174.h5 --port=34567 

#TRAIN RLaR_HF_1 port 35678
python scripts\ForagingExp.py --agent=DQN --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=5 --mdn-model-path=.\save\mdn2L_32x5x2_32x5x2_32x3x2_128_5_256_500_mse_0.0001_0.01202.h5 --port=45678 

#Need to run on Ubuntu side with the following command
#run argos on ubuntu correspondingly on port 23456, 40 robots, arena length of 6
bash ./src/ma_foraging/run/run.sh 199.17.162.13 10000 40 9
bash ./src/ma_foraging/run/run.sh 199.17.162.13 12345 40 9
bash ./src/ma_foraging/run/run.sh 199.17.162.13 23456 40 9
bash ./src/ma_foraging/run/run.sh 199.17.162.13 34567 40 9
bash ./src/ma_foraging/run/run.sh 199.17.162.13 45678 40 9

#RESTORE PREVIOUS TRAINING
python scripts\ForagingExp.py --agent=DQN --num-episodes=2099 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=2 --hidden-size=256 --lr=7.02e-05 --epsilon-start=0.21 --exploration-rate=0.8 --buffer-size=102400 --base-path=. --use-per --lr-decay --replay-interval=20 --norm-data --use-neighbor-image  --hidden-features-mode=0 --port=10000 --last-timesteps=901 --restore-log-path=./log/tf_board/DQN/FullHF_0.0001_325232523232_256_20210502_024854 --restore-model-path=./log/DQN/FullHF_0.0001_325232523232_256_20210502_024854.h5

python scripts\ForagingExp.py --agent=DQN --num-episodes=2197 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=2 --hidden-size=256 --lr=7.35e-05 --epsilon-start=0.249 --exploration-rate=0.8 --buffer-size=102400 --base-path=. --use-per --lr-decay --replay-interval=20 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=1 --mdn-model-path=.\save\mdn2L_32x5x2_32x5x2_32x3x2_32_1_256_500_mse_0.0001_0.01174.h5 --port=34567 --last-timesteps=803 --restore-log-path=./log/tf_board/DQN/mdnHF_0.0001_325232523232_256_20210502_025224 --restore-model-path=./log/DQN/mdnHF_0.0001_325232523232_256_20210502_025224.h5

#LSTM
#python .\scripts\ForagingExp.py --agent=LSTM --port=23456 --num-episodes=500 --steps-per-episode=150 --base-path=. --lstm-input-size=192 --num-batch-update-steps=2

#TEST
python scripts\ForagingExp.py --agent=DQN --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=40 --port=10000 

python scripts/ForagingExp.py --agent=DQN --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=./save/mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=./log/DQN/mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=20 --port=12345

bash ./src/ma_foraging/run/run.sh 127.0.0.1 10000 20 9

python scripts/ForagingExp.py --agent=DQN --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=./save/mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=./log/DQN/mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=20 --port=23456 
python scripts/ForagingExp.py --agent=DQN --port=10000 --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=./save/mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=./log/DQN/mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=20

python scripts/ForagingExp.py --agent=DQN --port=34567 --num-episodes=3000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=1.0 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=./save/mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=./log/DQN/mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=20

#test trained RLaR model with MDN 2 mixes with 75 robots
python scripts\ForagingExp.py --agent=DQN --port=10000 --num-episodes=1000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=75
#run argos on ubuntu correspondingly on port 10000 and 75 robots
bash ./src/ma_foraging/run/run.sh 127.0.0.1 10000 75

#test trained RLaR model with MDN 2 mixes with 80 robots on port 12345
python scripts\ForagingExp.py --agent=DQN --port=12345 --num-episodes=1000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=80
#run argos on ubuntu correspondingly on port 12345 and 80 robots
bash ./src/ma_foraging/run/run.sh 127.0.0.1 12345 80
 
python scripts\ForagingExp.py --agent=DQN --port=10000 --num-episodes=1000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=85

#run argos on ubuntu correspondingly on port 10000 and 85 robots
bash ./src/ma_foraging/run/run.sh 127.0.0.1 10000 85

python scripts\ForagingExp.py --agent=DQN --port=12345 --num-episodes=1000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=90

#run argos on ubuntu correspondingly on port 12345 and 90 robots
bash ./src/ma_foraging/run/run.sh 127.0.0.1 12345 90

#====================================================================
#test trained NoHF on port 23456 and 40 robots
python scripts\ForagingExp.py --agent=DQN --port=23456 --num-episodes=1000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=1 --restore-model-path=.\log\DQN\NoHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200529_2179eps_0.391_32.15.h5 --mode=test --num-robots=40

#run argos on ubuntu correspondingly on port 23456 and 40 robots
bash ./src/ma_foraging/run/run.sh 127.0.0.1 23456 40


#test trained RLaR 2 on port 12345, 40 robots, arena length of 7
python scripts\ForagingExp.py --agent=DQN --port=12345 --num-episodes=1000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=40

#run argos on ubuntu correspondingly on port 12345, 40 robots, arena length of 7
bash ./src/ma_foraging/run/run.sh 127.0.0.1 12345 40 7

#test trained RLaR 2 on port 12345, 40 robots, arena length of 6
python scripts\ForagingExp.py --agent=DQN --port=23456 --num-episodes=1000 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=40

#run argos on ubuntu correspondingly on port 23456, 40 robots, arena length of 6
bash ./src/ma_foraging/run/run.sh 127.0.0.1 23456 40 6

#test with GUI simulation
python scripts\ForagingExp.py --agent=DQN --port=12345 --num-episodes=10 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_32_5_2_32_5_2_32_3_2_256_20200606_2959eps_0.712_37.05.h5 --mode=test --num-robots=40

argos3 -c src/ma_foraging/argos_worlds/foraging.argos

#test on wsu machine with different arena length
#TEST
python scripts\ForagingExp.py --agent=DQN --num-episodes=3500 --steps-per-episode=150 --batch-size=64 --num-batch-update-steps=4 --hidden-size=256 --lr=0.0001 --epsilon-start=0.9999 --exploration-rate=0.8 --buffer-size=200000 --base-path=. --use-per --lr-decay --replay-interval=10 --norm-data --use-neighbor-image --hidden-features-mode=2 --n-mixes=2 --mdn-model-path=.\save\mdn_32x5x2_32x5x2_32x3x2_256_2_256_500_mse_0.0001_0.01286.h5 --restore-model-path=.\log\DQN\mdnHF_0.0001_325232523232_256_20201205_115104_2649eps_0.686_37.0.h5 --mode=test --num-robots=40 --port=10000 

bash ./src/ma_foraging/run/run.sh 199.17.162.13 10000 40 9